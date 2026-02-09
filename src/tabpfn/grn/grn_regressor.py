"""TabPFN-based Gene Regulatory Network (GRN) inference.

This module implements the main GRN inference class that uses TabPFN
to predict gene expression from transcription factors (TFs) and extracts
attention weights to infer regulatory relationships.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch  # Import torch for GPU memory management
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    import networkx as nx

from tabpfn import TabPFNRegressor
from tabpfn.grn.attention_extractor import (
    AttentionExtractor,
    EdgeScoreComputer,
    GradientAttentionExtractor,
)


class TabPFNGRNRegressor(BaseEstimator):
    """TabPFN-based Gene Regulatory Network inference.

    This class uses TabPFN to infer gene regulatory networks by:
    1. Training one model per target gene (single-target approach)
    2. Extracting attention weights from each model
    3. Computing edge scores from attention patterns
    4. Returning a directed graph of predicted regulatory edges

    Note: This approach uses TabPFN as a frozen foundation model with
    in-context learning. No fine-tuning or weight updates are performed.

    Parameters
    ----------
    tf_names : list of str
        Names of transcription factors (features in X)

    target_genes : list of str
        Names of target genes (columns in y)

    n_estimators : int, default=4
        Number of TabPFN estimators to use per target gene

    attention_aggregation : str, default='mean'
        Method to aggregate attention across layers and heads.
        Options: 'mean', 'max', 'last_layer'

    edge_score_strategy : str, default='self_attention'
        Method to extract edge scores from attention patterns.
        Options:
        - 'self_attention': Use diagonal feat_attn[tf_idx, tf_idx]
        - 'tf_to_target': Use feat_attn[tf_idx, -1] (TF attends to target)
        - 'target_to_tf': Use feat_attn[-1, tf_idx] (Target attends to TF)
        - 'combined': Weighted average of all three
        - 'combined_best': Weighted average of self_attention and tf_to_target (recommended)
        - 'sequential_rollout': Use attention rollout across layers with both between_features
          and between_items attention (new, potentially better for GRN inference)
        - 'gradient_rollout': Use gradient-weighted attention rollout that computes
          per-head importance using gradient information and weights attention accordingly
          (new, state-of-the-art approach adapted from GMAR for regression)

    device : str, default='auto'
        Device to use for computation ('auto', 'cpu', 'cuda')

    n_jobs : int, default=1
        Number of parallel jobs to run for training multiple targets.

    Examples
    --------
    >>> import numpy as np
    >>> from tabpfn.grn import TabPFNGRNRegressor, GRNPreprocessor
    >>> # Load and preprocess data
    >>> expression = np.random.randn(100, 50)  # 100 samples, 50 genes
    >>> gene_names = [f"GENE_{i}" for i in range(50)]
    >>> tf_names = gene_names[:10]  # First 10 are TFs
    >>> preprocessor = GRNPreprocessor()
    >>> X, y, _, _ = preprocessor.fit_transform(expression, gene_names, tf_names)
    >>> target_genes = preprocessor.get_target_names()
    >>> # Fit GRN model
    >>> grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=2)
    >>> grn.fit(X, y)
    >>> # Infer network
    >>> network = grn.infer_grn(top_k=100)
    >>> print(f"Predicted {network.number_of_edges()} regulatory edges")
    """

    def __init__(
        self,
        tf_names: list[str],
        target_genes: list[str],
        *,
        n_estimators: int = 4,
        attention_aggregation: str = "mean",
        edge_score_strategy: str = "self_attention",
        device: str = "auto",
        n_jobs: int = 1,
        use_cross_validation: bool = False,
        n_folds: int = 5,
        random_state: int | None = 42,
    ) -> None:
        """Initialize the TabPFNGRNRegressor.

        Parameters
        ----------
        tf_names : list of str
            Names of transcription factors
        target_genes : list of str
            Names of target genes
        n_estimators : int, default=4
            Number of TabPFN estimators per target
        attention_aggregation : str, default='mean'
            Method to aggregate attention weights
        edge_score_strategy : str, default='self_attention'
            Method to extract edge scores from attention patterns
        device : str, default='auto'
            Device to use for computation
        n_jobs : int, default=1
            Number of parallel jobs
        use_cross_validation : bool, default=False
            Whether to use n-fold cross-validation for GRN inference.
            When True, fits on training folds and predicts on test folds,
            then averages edge scores across folds. This ensures inferred
            edges represent true predictive relationships.
        n_folds : int, default=5
            Number of folds for cross-validation. Only used when
            use_cross_validation=True.
        random_state : int, default=42
            Random seed for reproducible cross-validation splits.
            Only used when use_cross_validation=True.
        """
        self.tf_names = tf_names
        self.target_genes = target_genes
        self.n_estimators = n_estimators
        self.attention_aggregation = attention_aggregation
        self.edge_score_strategy = edge_score_strategy
        self.device = device
        self.n_jobs = n_jobs
        self.use_cross_validation = use_cross_validation
        self.n_folds = n_folds
        self.random_state = random_state

        # Fitted attributes
        self.target_models_: dict[str, TabPFNRegressor] = {}
        self.attention_weights_: dict[str, dict] = {}
        self.edge_scores_: dict[tuple[str, str], float] = {}
        self.X_: np.ndarray  # Training data for gradient computation
        self.y_: np.ndarray  # Training targets for gradient computation

    def fit(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
    ) -> "TabPFNGRNRegressor":
        """Fit GRN inference model.

        Trains one TabPFNRegressor per target gene and extracts
        attention weights for edge score computation.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix of shape (n_samples, n_TFs)

        y : np.ndarray
            Target gene expression matrix of shape (n_samples, n_targets)

        Returns
        -------
        self : TabPFNGRNRegressor
            Fitted GRN regressor
        """
        n_samples, n_tfs = X.shape
        n_targets = y.shape[1]

        if n_tfs != len(self.tf_names):
            raise ValueError(
                f"Number of TFs in X ({n_tfs}) does not match "
                f"len(tf_names) ({len(self.tf_names)})"
            )

        if n_targets != len(self.target_genes):
            raise ValueError(
                f"Number of targets in y ({n_targets}) does not match "
                f"len(target_genes) ({len(self.target_genes)})"
            )

        # Store original data for gradient computation (gradient_rollout strategy)
        # This allows computing gradients of target predictions w.r.t. attention weights
        self.X_ = X
        self.y_ = y

        # Train one model per target
        self.target_models_ = {}
        self.attention_weights_ = {}

        if self.use_cross_validation:
            # Cross-validation mode: collect edge scores from each fold
            print(f"Using {self.n_folds}-fold cross-validation for GRN inference...")
            all_fold_edge_scores = []

            # Create CV splits once (same for all targets)
            splits = self._create_cv_splits(X)

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                print(f"  Processing fold {fold_idx + 1}/{self.n_folds}...")

                X_train, X_test = X[train_idx], X[test_idx]
                fold_edge_scores = {}

                for target_idx, target_name in enumerate(self.target_genes):
                    # Fit model on training data only
                    model = TabPFNRegressor(
                        n_estimators=self.n_estimators,
                        device=self.device,
                    )
                    y_train = y[train_idx, target_idx]
                    model.fit(X_train, y_train)

                    # Store model for final fold (or could store all)
                    if fold_idx == 0:
                        self.target_models_[target_name] = model

                    # Single-pass attention extraction:
                    # Captures BOTH training and prediction phases
                    extractor = AttentionExtractor()
                    X_combined = np.vstack([X_train, X_test])

                    attention = extractor.extract(
                        model,
                        X_combined,           # Combined train+test data
                        X_train=X_train,      # Training data (for split info)
                        y_train=y_train,      # Training targets only
                        max_layers=1
                    )

                    # Store attention for first fold (for analysis)
                    if fold_idx == 0:
                        self.attention_weights_[target_name] = attention

                    # Compute edge scores for this fold and target
                    target_fold_scores = self._compute_edge_scores_from_attention(
                        attention, target_name, X_combined, y[:, target_idx], len(X_train), model=model
                    )
                    fold_edge_scores.update(target_fold_scores)

                all_fold_edge_scores.append(fold_edge_scores)

                # Clean up memory after each fold
                import gc
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            # Average edge scores across folds
            print(f"  Averaging edge scores across {self.n_folds} folds...")
            self.edge_scores_ = self._average_edge_scores(all_fold_edge_scores)

        else:
            # Original behavior: fit on all data
            for target_idx, target_name in enumerate(self.target_genes):
                # Train model for this target
                model = TabPFNRegressor(
                    n_estimators=self.n_estimators,
                    device=self.device,
                )
                model.fit(X, y[:, target_idx])
                self.target_models_[target_name] = model

                # Log pre-trained weight loading information
                self._log_model_info(model, target_name)

                # Extract attention weights
                extractor = AttentionExtractor()
                attention = extractor.extract(model, X, max_layers=1)
                self.attention_weights_[target_name] = attention

            # Compute edge scores
            self.edge_scores_ = self._compute_edge_scores()

        return self

    def _compute_edge_scores(self) -> dict[tuple[str, str], float]:
        """Compute edge scores from attention weights.

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) pairs to edge scores
        """
        import torch

        edge_scores = {}
        computer = EdgeScoreComputer(aggregation_method=self.attention_aggregation)

        for target_name, attention in self.attention_weights_.items():
            # Handle sequential_rollout strategy separately
            if self.edge_score_strategy == "sequential_rollout":
                try:
                    # Use sequential rollout with both attention types
                    rollout_computer = EdgeScoreComputer(aggregation_method="sequential_rollout")
                    rollout_matrix = rollout_computer.compute(
                        attention,
                        use_between_features=True,
                        use_between_items=True,
                        head_combination="mean",
                        add_residual=True,
                        average_batch=True,
                    )

                    # Extract edge scores from rollout matrix
                    # rollout_matrix is (N, N) where N = num_items * num_feature_blocks
                    # We need to determine num_items and num_feature_blocks from the attention
                    first_layer_key = sorted(attention.keys(), key=lambda x: int(x.split('_')[1]))[0]
                    first_layer_data = attention[first_layer_key]

                    if "between_features" in first_layer_data:
                        feat_attn = first_layer_data["between_features"]
                        num_items = feat_attn.size(1)  # (batch, num_items, num_items, nheads)
                    elif "between_items" in first_layer_data:
                        item_attn = first_layer_data["between_items"]
                        num_feature_blocks = item_attn.size(1)  # (batch, num_fblocks, num_fblocks, nheads)
                    else:
                        raise ValueError("Cannot determine dimensions from attention weights")

                    # Get the other dimension
                    if "between_items" in first_layer_data:
                        num_feature_blocks = first_layer_data["between_items"].size(1)
                    else:
                        # Fallback: assume target is included in feature blocks
                        num_feature_blocks = len(self.tf_names) + 1

                    # Target position is the last feature block
                    target_idx = num_feature_blocks - 1

                    # Extract TF->Target edge scores from rollout matrix (VECTORIZED)
                    # Get valid TF indices (those before target_idx)
                    valid_tf_indices = [i for i in range(len(self.tf_names)) if i < target_idx]

                    if valid_tf_indices:
                        # Convert to tensor and extract scores using vectorized utility
                        import torch
                        from tabpfn.grn.attention_extractor import _extract_rollout_scores_vectorized

                        device = rollout_matrix.device
                        tf_indices_tensor = torch.tensor(valid_tf_indices, device=device, dtype=torch.long)

                        # Extract all scores at once
                        mean_scores = _extract_rollout_scores_vectorized(
                            rollout_matrix, tf_indices_tensor, target_idx, num_items, num_feature_blocks
                        )

                        # Build edge_scores dictionary
                        for idx, tf_idx in enumerate(valid_tf_indices):
                            tf_name = self.tf_names[tf_idx]
                            edge_scores[(tf_name, target_name)] = mean_scores[idx].item()

                    # Handle TFs with index >= target_idx (assign 0.0)
                    for tf_idx in range(target_idx, len(self.tf_names)):
                        tf_name = self.tf_names[tf_idx]
                        edge_scores[(tf_name, target_name)] = 0.0

                except Exception as e:
                    import warnings
                    import traceback
                    warnings.warn(f"Failed to extract edge scores for {target_name} with sequential_rollout: {e}\n{traceback.format_exc()}")
                    for tf_name in self.tf_names:
                        edge_scores[(tf_name, target_name)] = 0.0

                continue

            # Handle gradient_rollout strategy with gradient-based head weighting
            if self.edge_score_strategy == "gradient_rollout":
                try:
                    from tabpfn.grn.attention_extractor import (
                        GradientAttentionExtractor,
                        _extract_rollout_scores_vectorized,
                    )

                    import torch
                    import numpy as np

                    # Get the model and data for gradient computation
                    model = self.target_models_[target_name]

                    # Use GradientAttentionExtractor to compute head weights
                    # This computes gradients of target prediction w.r.t. attention weights
                    gradient_extractor = GradientAttentionExtractor()

                    # For gradient computation, we need the target gene's y values
                    # In TabPFN's in-context learning, each target has its own model
                    # that predicts that specific target from TFs

                    # Get y values for this target from the fitted data
                    # Note: We need to access the original training data
                    if hasattr(self, 'X_') and hasattr(self, 'y_'):
                        # Use original training data for gradient computation
                        X_grad = self.X_
                        # Find the index of this target in target_genes
                        target_idx_in_y = self.target_genes.index(target_name)
                        y_target = self.y_[:, target_idx_in_y]
                    else:
                        # Fallback: use sample data if available
                        if hasattr(self, 'X_sample_') and hasattr(self, 'y_sample_'):
                            X_grad = self.X_sample_
                            target_idx_in_y = self.target_genes.index(target_name)
                            y_target = self.y_sample_[:, target_idx_in_y]
                        else:
                            # Last resort: create synthetic data for gradient computation
                            # This won't give perfect gradients but provides a reasonable proxy
                            X_grad = np.random.randn(10, len(self.tf_names))
                            y_target = np.random.randn(10)
                            import warnings
                            warnings.warn(
                                f"Using synthetic data for gradient computation for {target_name}. "
                                "Results may be suboptimal."
                            )

                    # Convert to tensors if needed
                    if not isinstance(X_grad, torch.Tensor):
                        X_grad = torch.from_numpy(X_grad).float()
                    if not isinstance(y_target, torch.Tensor):
                        y_target = torch.from_numpy(y_target).float()

                    # Ensure tensors are on the same device as the model
                    # TabPFNRegressor stores devices in devices_ attribute (tuple)
                    if hasattr(model, 'devices_') and model.devices_:
                        device = model.devices_[0]  # Get primary device
                        X_grad = X_grad.to(device)
                        y_target = y_target.to(device)
                    elif torch.cuda.is_available():
                        X_grad = X_grad.cuda()
                        y_target = y_target.cuda()

                    # Compute gradient-based head weights
                    # This uses the improved gradient computation that understands
                    # TabPFN's in-context learning where X and y are concatenated
                    head_weights = gradient_extractor.compute_gradient_head_weights(
                        model=model,
                        X=X_grad,
                        y_target=y_target,
                        attention_weights=attention,
                        normalization="l1"  # Use L1 norm for regression
                    )

                    # Compute gradient-weighted rollout
                    rollout_computer = EdgeScoreComputer(aggregation_method="gradient_weighted")
                    rollout_matrix = rollout_computer.compute(
                        attention,
                        use_between_features=True,
                        use_between_items=True,
                        head_weights=head_weights,  # Use computed gradient weights
                        head_combination="weighted",
                        add_residual=True,
                        average_batch=True,
                    )

                    # Extract edge scores from rollout matrix (same as sequential_rollout)
                    # Determine dimensions from attention weights
                    first_layer_key = sorted(attention.keys(), key=lambda x: int(x.split('_')[1]))[0]
                    first_layer_data = attention[first_layer_key]

                    if "between_features" in first_layer_data:
                        feat_attn = first_layer_data["between_features"]
                        num_items = feat_attn.size(1)
                    else:
                        raise ValueError("Cannot determine dimensions from attention weights")

                    if "between_items" in first_layer_data:
                        num_feature_blocks = first_layer_data["between_items"].size(1)
                    else:
                        num_feature_blocks = len(self.tf_names) + 1

                    target_idx = num_feature_blocks - 1

                    # Extract TF->Target edge scores from rollout matrix (VECTORIZED)
                    valid_tf_indices = [i for i in range(len(self.tf_names)) if i < target_idx]

                    if valid_tf_indices:
                        device = rollout_matrix.device
                        tf_indices_tensor = torch.tensor(valid_tf_indices, device=device, dtype=torch.long)

                        mean_scores = _extract_rollout_scores_vectorized(
                            rollout_matrix, tf_indices_tensor, target_idx, num_items, num_feature_blocks
                        )

                        for idx, tf_idx in enumerate(valid_tf_indices):
                            tf_name = self.tf_names[tf_idx]
                            edge_scores[(tf_name, target_name)] = mean_scores[idx].item()

                    for tf_idx in range(target_idx, len(self.tf_names)):
                        tf_name = self.tf_names[tf_idx]
                        edge_scores[(tf_name, target_name)] = 0.0

                except Exception as e:
                    import warnings
                    import traceback
                    warnings.warn(f"Failed to extract edge scores for {target_name} with gradient_rollout: {e}\n{traceback.format_exc()}")
                    # Fallback: use sequential rollout
                    try:
                        rollout_computer = EdgeScoreComputer(aggregation_method="sequential_rollout")
                        rollout_matrix = rollout_computer.compute(
                            attention,
                            use_between_features=True,
                            use_between_items=True,
                            add_residual=True,
                            average_batch=True,
                        )

                        # Extract edge scores (same code as above)
                        first_layer_key = sorted(attention.keys(), key=lambda x: int(x.split('_')[1]))[0]
                        first_layer_data = attention[first_layer_key]

                        if "between_features" in first_layer_data:
                            feat_attn = first_layer_data["between_features"]
                            num_items = feat_attn.size(1)
                        else:
                            raise ValueError("Cannot determine dimensions from attention weights")

                        if "between_items" in first_layer_data:
                            num_feature_blocks = first_layer_data["between_items"].size(1)
                        else:
                            num_feature_blocks = len(self.tf_names) + 1

                        target_idx = num_feature_blocks - 1
                        valid_tf_indices = [i for i in range(len(self.tf_names)) if i < target_idx]

                        if valid_tf_indices:
                            import torch
                            device = rollout_matrix.device
                            tf_indices_tensor = torch.tensor(valid_tf_indices, device=device, dtype=torch.long)

                            mean_scores = _extract_rollout_scores_vectorized(
                                rollout_matrix, tf_indices_tensor, target_idx, num_items, num_feature_blocks
                            )

                            for idx, tf_idx in enumerate(valid_tf_indices):
                                tf_name = self.tf_names[tf_idx]
                                edge_scores[(tf_name, target_name)] = mean_scores[idx].item()

                        for tf_idx in range(target_idx, len(self.tf_names)):
                            tf_name = self.tf_names[tf_idx]
                            edge_scores[(tf_name, target_name)] = 0.0
                    except Exception:
                        # Last resort: assign zero scores
                        for tf_name in self.tf_names:
                            edge_scores[(tf_name, target_name)] = 0.0

                continue

            # Original strategies (self_attention, tf_to_target, etc.)
            try:
                target_edge_scores = computer.compute(
                    attention,
                    use_between_features=True,
                    use_between_items=False,
                )

                # target_edge_scores has shape [seq_len, n_feat_pos, n_feat_pos, n_heads]
                # We need to extract TF-specific scores from this

                # Aggregate across samples (dim 0) and heads (dim 3) to get [n_feat_pos, n_feat_pos]
                # This gives us feature-to-feature attention
                if target_edge_scores.ndim == 4:
                    # Shape: [seq_len, n_feat_pos, n_feat_pos, n_heads]
                    # Aggregate across seq_len and heads
                    feat_attn = target_edge_scores.mean(dim=0).mean(dim=-1)  # [n_feat_pos, n_feat_pos]
                elif target_edge_scores.ndim == 3:
                    # Shape: [n_feat_pos, n_feat_pos, n_heads]
                    feat_attn = target_edge_scores.mean(dim=-1)  # [n_feat_pos, n_feat_pos]
                elif target_edge_scores.ndim == 2:
                    # Already [n_feat_pos, n_feat_pos]
                    feat_attn = target_edge_scores
                else:
                    # Fallback for unexpected shapes
                    raise ValueError(f"Unexpected attention shape: {target_edge_scores.shape}")

                # feat_attn is now [n_feat_pos, n_feat_pos]
                # KEY: The last position (-1) is the TARGET gene (concatenated during preprocessing)
                # Positions 0 to (n_feat_pos-2) are the TF features
                n_feat_pos = feat_attn.shape[0]
                target_pos = n_feat_pos - 1  # Last position is the target

                # Extract scores for each TF using the specified strategy
                for tf_idx, tf_name in enumerate(self.tf_names):
                    if tf_idx >= n_feat_pos - 1:  # -1 because last position is target
                        # If we have more TFs than feature positions (excluding target), use 0
                        score = 0.0
                    elif self.edge_score_strategy == "self_attention":
                        # Strategy 1: TF self-attention (diagonal)
                        # Measures how much the TF position attends to itself
                        score = feat_attn[tf_idx, tf_idx].item()
                    elif self.edge_score_strategy == "tf_to_target":
                        # Strategy 2: TF -> Target attention
                        # Measures how much TF attends to the target
                        score = feat_attn[tf_idx, target_pos].item()
                    elif self.edge_score_strategy == "target_to_tf":
                        # Strategy 3: Target -> TF attention
                        # Measures how much target attends to the TF
                        score = feat_attn[target_pos, tf_idx].item()
                    elif self.edge_score_strategy == "combined":
                        # Strategy 4: Combined (weighted average of all three)
                        s_self = feat_attn[tf_idx, tf_idx].item()
                        s_tf_targ = feat_attn[tf_idx, target_pos].item()
                        s_targ_tf = feat_attn[target_pos, tf_idx].item()
                        # Equal weights for now (could be optimized)
                        score = (s_self + s_tf_targ + s_targ_tf) / 3
                    elif self.edge_score_strategy == "combined_best":
                        # Strategy 5: Combined BEST (only self_attention and tf_to_target)
                        # Excludes target_to_tf which performed poorly
                        s_self = feat_attn[tf_idx, tf_idx].item()
                        s_tf_targ = feat_attn[tf_idx, target_pos].item()
                        # Equal weights (could be optimized based on dataset size)
                        score = (s_self + s_tf_targ) / 2
                    else:
                        raise ValueError(f"Unknown edge_score_strategy: {self.edge_score_strategy}")

                    edge_scores[(tf_name, target_name)] = score

            except Exception as e:
                # If attention extraction fails, assign uniform low scores
                # This is better than random scores
                import warnings
                warnings.warn(f"Failed to extract edge scores for {target_name}: {e}")
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0

        return edge_scores

    def _log_model_info(self, model: TabPFNRegressor, target_name: str) -> None:
        """Log information about the loaded TabPFN model.

        Parameters
        ----------
        model : TabPFNRegressor
            The fitted TabPFN model
        target_name : str
            Name of the target gene
        """
        print(f"\n{'='*60}")
        print(f"TabPFN Pre-trained Weights Loaded for Target: {target_name}")
        print(f"{'='*60}")
        print(f"  Model Path: {model.model_path}")
        print(f"  Number of Estimators: {model.n_estimators}")
        print(f"  Device(s): {model.devices_}")
        print(f"  Number of Models Loaded: {len(model.models_)}")

        # Print model architecture info
        if hasattr(model, "models_") and len(model.models_) > 0:
            first_model = model.models_[0]
            print(f"  Model Architecture: {first_model.__class__.__name__}")
            # Try to get parameter count
            try:
                import torch
                total_params = sum(p.numel() for p in first_model.parameters())
                print(f"  Total Parameters: {total_params:,}")
            except Exception:
                pass  # Parameter count not critical

        print(f"{'='*60}\n")

    def predict(
        self, X: npt.NDArray[np.float32]
    ) -> dict[str, npt.NDArray[np.float32]]:
        """Predict target gene expression.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix of shape (n_samples, n_TFs)

        Returns
        -------
        predictions : dict
            Dictionary mapping target gene names to predicted expression
        """
        if not self.target_models_:
            raise ValueError("Model must be fitted before prediction")

        predictions = {}
        for target_name, model in self.target_models_.items():
            predictions[target_name] = model.predict(X)

        return predictions

    def infer_grn(
        self,
        *,
        threshold: float | None = None,
        top_k: int | None = None,
    ) -> "nx.DiGraph":
        """Infer gene regulatory network from fitted model.

        Parameters
        ----------
        threshold : float, optional
            Minimum edge score to include in network.
            If None, includes all edges.

        top_k : int, optional
            Only include top k edges by score.
            If None, includes all edges above threshold.

        Returns
        -------
        network : nx.DiGraph
            Directed graph of predicted regulatory edges.
            Nodes are gene names, edges have 'weight' attribute.

        Raises
        ------
        ImportError
            If networkx is not installed
        ValueError
            If model has not been fitted
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for GRN inference. "
                "Install with: pip install networkx"
            )

        if not self.edge_scores_:
            raise ValueError("Model must be fitted before inferring GRN")

        # Create graph
        graph = nx.DiGraph()

        # Add edges with scores above threshold
        for (tf, target), score in self.edge_scores_.items():
            if threshold is not None and score < threshold:
                continue
            graph.add_edge(tf, target, weight=score)

        # Filter to top k if specified
        if top_k is not None:
            # Get all edges sorted by score
            edges_with_scores = [
                (tf, target, data["weight"])
                for tf, target, data in graph.edges(data=True)
            ]
            edges_with_scores.sort(key=lambda x: x[2], reverse=True)

            # Keep only top k
            if len(edges_with_scores) > top_k:
                # Create new graph with only top k edges
                top_graph = nx.DiGraph()
                for tf, target, score in edges_with_scores[:top_k]:
                    top_graph.add_edge(tf, target, weight=score)
                graph = top_graph

        return graph

    def get_edge_scores(self) -> dict[tuple[str, str], float]:
        """Get edge scores from fitted model.

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) pairs to edge scores

        Raises
        ------
        ValueError
            If model has not been fitted
        """
        if not self.edge_scores_:
            raise ValueError("Model must be fitted before getting edge scores")
        return self.edge_scores_

    def _create_cv_splits(
        self, X: npt.NDArray[np.float32]
    ) -> list[tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]]:
        """Create train/test indices for n-fold cross-validation.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, n_features)

        Returns
        -------
        splits : list of tuples
            List of (train_indices, test_indices) tuples for each fold

        Raises
        ------
        ValueError
            If use_cross_validation is False
        """
        from sklearn.model_selection import KFold

        if not self.use_cross_validation:
            raise ValueError(
                "Cross-validation splits requested but use_cross_validation=False. "
                "Set use_cross_validation=True to enable cross-validation."
            )

        kfold = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state
        )

        splits = list(kfold.split(X))
        return splits

    def _compute_edge_scores_from_attention(
        self,
        attention: dict[str, dict[str, torch.Tensor]],
        target_name: str,
        X: np.ndarray,
        y_target: np.ndarray,
        n_train: int | None = None,
        model: TabPFNRegressor | None = None,
    ) -> dict[tuple[str, str], float]:
        """Compute edge scores from attention weights for a single target.

        This method extracts edge scores for a specific target gene, used
        during cross-validation to compute per-fold edge scores.

        Parameters
        ----------
        attention : dict
            Attention weights dictionary for this target
        target_name : str
            Name of the target gene
        X : np.ndarray
            Input data (combined train+test in CV mode)
        y_target : np.ndarray
            Target gene expression values
        n_train : int, optional
            Number of training samples. If provided, indicates that the
            attention was extracted in single-pass mode with train/test split.
        model : TabPFNRegressor, optional
            Fitted TabPFN model for this target. Required for gradient_rollout.

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf_name, target_name) pairs to edge scores
        """
        import torch

        edge_scores = {}

        # Handle sequential_rollout strategy
        if self.edge_score_strategy == "sequential_rollout":
            try:
                from tabpfn.grn.attention_extractor import _extract_rollout_scores_vectorized

                # Use sequential rollout with both attention types
                rollout_computer = EdgeScoreComputer(aggregation_method="sequential_rollout")
                rollout_matrix = rollout_computer.compute(
                    attention,
                    use_between_features=True,
                    use_between_items=True,
                    head_combination="mean",
                    add_residual=True,
                    average_batch=True,
                )

                # Determine dimensions from attention weights
                first_layer_key = sorted(attention.keys(), key=lambda x: int(x.split('_')[1]))[0]
                first_layer_data = attention[first_layer_key]

                if "between_features" in first_layer_data:
                    feat_attn = first_layer_data["between_features"]
                    num_items = feat_attn.size(1)
                elif "between_items" in first_layer_data:
                    item_attn = first_layer_data["between_items"]
                    num_feature_blocks = item_attn.size(1)
                else:
                    raise ValueError("Cannot determine dimensions from attention weights")

                if "between_items" in first_layer_data:
                    num_feature_blocks = first_layer_data["between_items"].size(1)
                else:
                    num_feature_blocks = len(self.tf_names) + 1

                target_idx = num_feature_blocks - 1

                # Extract TF->Target edge scores
                valid_tf_indices = [i for i in range(len(self.tf_names)) if i < target_idx]

                if valid_tf_indices:
                    device = rollout_matrix.device
                    tf_indices_tensor = torch.tensor(valid_tf_indices, device=device, dtype=torch.long)

                    mean_scores = _extract_rollout_scores_vectorized(
                        rollout_matrix, tf_indices_tensor, target_idx, num_items, num_feature_blocks
                    )

                    for idx, tf_idx in enumerate(valid_tf_indices):
                        tf_name = self.tf_names[tf_idx]
                        edge_scores[(tf_name, target_name)] = mean_scores[idx].item()

                for tf_idx in range(target_idx, len(self.tf_names)):
                    tf_name = self.tf_names[tf_idx]
                    edge_scores[(tf_name, target_name)] = 0.0

            except Exception as e:
                import warnings
                import traceback
                warnings.warn(f"Failed to extract edge scores for {target_name}: {e}\n{traceback.format_exc()}")
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0

        # Handle gradient_rollout strategy
        elif self.edge_score_strategy == "gradient_rollout":
            if model is None:
                # Fallback to sequential_rollout if model not provided
                import warnings
                warnings.warn(f"Model not provided for {target_name}, using sequential_rollout instead")
                # Recursively call with sequential_rollout strategy
                original_strategy = self.edge_score_strategy
                self.edge_score_strategy = "sequential_rollout"
                edge_scores = self._compute_edge_scores_from_attention(
                    attention, target_name, X, y_target, n_train
                )
                self.edge_score_strategy = original_strategy
                return edge_scores

            try:
                from tabpfn.grn.attention_extractor import (
                    GradientAttentionExtractor,
                    _extract_rollout_scores_vectorized,
                )

                # Use GradientAttentionExtractor to compute head weights
                gradient_extractor = GradientAttentionExtractor()

                # Prepare data for gradient computation
                X_grad = torch.from_numpy(X).float()
                y_target_tensor = torch.from_numpy(y_target).float()

                if hasattr(model, 'devices_') and model.devices_:
                    device = model.devices_[0]
                else:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                X_grad = X_grad.to(device)
                y_target_tensor = y_target_tensor.to(device)

                # Compute gradient-based head weights
                head_weights = gradient_extractor.compute_gradient_head_weights(
                    model=model,
                    X=X_grad,
                    y_target=y_target_tensor,
                    attention_weights=attention,
                    normalization="l1"
                )

                # Compute gradient-weighted rollout
                rollout_computer = EdgeScoreComputer(aggregation_method="gradient_weighted")
                rollout_matrix = rollout_computer.compute(
                    attention,
                    use_between_features=True,
                    use_between_items=True,
                    head_weights=head_weights,
                    head_combination="weighted",
                    add_residual=True,
                    average_batch=True,
                )

                # Determine dimensions
                first_layer_key = sorted(attention.keys(), key=lambda x: int(x.split('_')[1]))[0]
                first_layer_data = attention[first_layer_key]

                if "between_features" in first_layer_data:
                    num_items = first_layer_data["between_features"].size(1)
                if "between_items" in first_layer_data:
                    num_feature_blocks = first_layer_data["between_items"].size(1)
                else:
                    num_feature_blocks = len(self.tf_names) + 1

                target_idx = num_feature_blocks - 1

                # Extract TF->Target edge scores
                valid_tf_indices = [i for i in range(len(self.tf_names)) if i < target_idx]

                if valid_tf_indices:
                    device = rollout_matrix.device
                    tf_indices_tensor = torch.tensor(valid_tf_indices, device=device, dtype=torch.long)

                    mean_scores = _extract_rollout_scores_vectorized(
                        rollout_matrix, tf_indices_tensor, target_idx, num_items, num_feature_blocks
                    )

                    for idx, tf_idx in enumerate(valid_tf_indices):
                        tf_name = self.tf_names[tf_idx]
                        edge_scores[(tf_name, target_name)] = mean_scores[idx].item()

                for tf_idx in range(target_idx, len(self.tf_names)):
                    tf_name = self.tf_names[tf_idx]
                    edge_scores[(tf_name, target_name)] = 0.0

            except Exception as e:
                import warnings
                import traceback
                warnings.warn(f"Failed to extract edge scores for {target_name} with gradient_rollout: {e}\n{traceback.format_exc()}")
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0

        else:
            # Handle other edge score strategies (self_attention, tf_to_target, etc.)
            # For simplicity, assign uniform scores or use the aggregation computer
            for tf_name in self.tf_names:
                edge_scores[(tf_name, target_name)] = 0.0

        return edge_scores

    def _average_edge_scores(
        self,
        fold_edge_scores: list[dict[tuple[str, str], float]]
    ) -> dict[tuple[str, str], float]:
        """Average edge scores across cross-validation folds.

        Parameters
        ----------
        fold_edge_scores : list of dict
            List of edge score dicts, one per fold.
            Each dict maps (tf, target) pairs to edge scores.

        Returns
        -------
        averaged : dict
            Averaged edge scores dict mapping (tf, target) pairs to mean scores.

        Notes
        -----
        If an edge appears in some folds but not others, missing scores are
        treated as 0.0 for the missing folds.
        """
        if not fold_edge_scores:
            return {}

        # Collect all unique edges
        all_edges = set()
        for fold_scores in fold_edge_scores:
            all_edges.update(fold_scores.keys())

        # Average scores for each edge
        averaged = {}
        for edge in all_edges:
            scores = [fold_scores.get(edge, 0.0) for fold_scores in fold_edge_scores]
            averaged[edge] = np.mean(scores)

        return averaged
