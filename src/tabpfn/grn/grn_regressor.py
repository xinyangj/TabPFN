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
    compute_kronecker_rollout_scores,
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
        - 'integrated_gradients': Use Integrated Gradients (Sundararajan et al., 2017) to
          compute feature attributions via gradient interpolation from a zero baseline.
          Captures causal feature importance rather than attention routing patterns.
        - 'rise': Use RISE (Randomized Input Sampling for Explanation) to compute
          feature attributions via random binary masking. Forward-only (no gradients),
          avoids gradient coupling through TabPFN's grouped-feature architecture.

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
        use_kronecker: bool = True,
        ig_n_folds: int = 1,
        ig_baseline: str = "zero",
        rise_n_masks: int = 500,
        rise_mask_prob: float = 0.5,
        rise_baseline: str = "zero",
        rise_n_folds: int = 1,
        shapley_n_permutations: int = 200,
        shapley_n_folds: int = 1,
        shapley_exact_threshold: int = 15,
        shapley_method: str = "auto",
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
        use_kronecker : bool, default=True
            Whether to use memory-efficient Kronecker-factored rollout for
            sequential_rollout and gradient_rollout strategies. When True,
            avoids building the full N×N rollout matrix (O(M²F) instead of
            O(M²F²) memory). Set to False to use the original full-matrix
            rollout. Only affects rollout-based strategies.
        ig_n_folds : int, default=1
            Number of cross-validation folds for Integrated Gradients.
            When >1, rotates train/test splits and averages per-TF scores.
            Only used when edge_score_strategy='integrated_gradients'.
        ig_baseline : str, default='zero'
            Baseline for Integrated Gradients interpolation.
            ``'zero'``: all-zero vector (no TF expression).
            ``'mean'``: per-feature mean of training samples in each fold.
            Only used when edge_score_strategy='integrated_gradients'.
        rise_n_masks : int, default=500
            Number of random binary masks for RISE.
            Only used when edge_score_strategy='rise'.
        rise_mask_prob : float, default=0.5
            Probability of keeping each feature in RISE masks.
            Only used when edge_score_strategy='rise'.
        rise_baseline : str, default='zero'
            Fill value for masked-out features in RISE.
            ``'zero'``: fill with zeros. ``'mean'``: fill with training mean.
            Only used when edge_score_strategy='rise'.
        rise_n_folds : int, default=1
            Number of cross-validation folds for RISE.
            When >1, rotates train/test splits and averages per-TF scores.
            Only used when edge_score_strategy='rise'.
        shapley_n_permutations : int, default=200
            Number of random permutations for approximate Shapley values.
            Only used when n_TFs > shapley_exact_threshold.
            Only used when edge_score_strategy='shapley'.
        shapley_n_folds : int, default=1
            Number of cross-validation folds for Shapley.
            When >1, rotates train/test splits and averages per-TF scores.
            Only used when edge_score_strategy='shapley'.
        shapley_exact_threshold : int, default=15
            Use exact Shapley (all 2^n coalitions) when n_TFs ≤ this value.
            Otherwise use permutation-based approximation.
            Only used when edge_score_strategy='shapley'.
        shapley_method : str, default='auto'
            Shapley computation method. Options:
            ``'auto'``: exact (≤threshold TFs) or permutation (>threshold).
            ``'kernelshap_test'``: KernelSHAP masking test features only.
            ``'kernelshap_train'``: KernelSHAP masking train features only
            (captures in-context learning signal).
            ``'kernelshap_full'``: KernelSHAP masking both train and test.
            Only used when edge_score_strategy='shapley'.
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
        self.use_kronecker = use_kronecker
        self.ig_n_folds = ig_n_folds
        self.ig_baseline = ig_baseline
        self.rise_n_masks = rise_n_masks
        self.rise_mask_prob = rise_mask_prob
        self.rise_baseline = rise_baseline
        self.rise_n_folds = rise_n_folds
        self.shapley_n_permutations = shapley_n_permutations
        self.shapley_n_folds = shapley_n_folds
        self.shapley_exact_threshold = shapley_exact_threshold
        self.shapley_method = shapley_method

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
        shared_model_arch=None,
        shared_device=None,
        shared_criterion=None,
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

        shared_model_arch : torch.nn.Module, optional
            Pre-loaded TabPFN architecture to reuse (avoids redundant weight
            loading). Used for integrated_gradients and rise strategies.
        shared_device : torch.device, optional
            Device for the shared model arch.

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
            if self.edge_score_strategy in ("integrated_gradients", "rise", "shapley"):
                # IG/RISE/Shapley only need the pre-trained architecture (same for all
                # targets) — load it once and compute directly.
                if shared_model_arch is not None:
                    self._shared_model_arch_ = shared_model_arch
                    self._shared_device_ = shared_device
                    self._shared_criterion_ = shared_criterion
                else:
                    model = TabPFNRegressor(
                        n_estimators=self.n_estimators,
                        device=self.device,
                    )
                    # fit on first target just to trigger weight loading
                    model.fit(X, y[:, 0])
                    self._log_model_info(model, self.target_genes[0])
                    self._shared_model_arch_ = model.models_[0]
                    self._shared_criterion_ = model.znorm_space_bardist_
                    if hasattr(model, "devices_") and model.devices_:
                        self._shared_device_ = model.devices_[0]
                    else:
                        self._shared_device_ = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu"
                        )
            else:
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
                    # Use all layers for rollout strategies; 1 layer otherwise
                    use_all_layers = self.edge_score_strategy in (
                        "gradient_rollout", "sequential_rollout",
                    )
                    attention = extractor.extract(
                        model, X, max_layers=None if use_all_layers else 1
                    )
                    self.attention_weights_[target_name] = attention

            # Compute edge scores
            self.edge_scores_ = self._compute_edge_scores()

        return self

    def _rollout_edge_scores(
        self,
        attention: dict[str, dict[str, torch.Tensor]],
        target_name: str,
        head_weights: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> dict[tuple[str, str], float]:
        """Compute rollout-based edge scores.

        Uses either memory-efficient Kronecker factorisation or the full
        N×N rollout matrix depending on ``self.use_kronecker``.

        Parameters
        ----------
        attention : dict
            Per-layer attention weights for one target.
        target_name : str
            Name of the target gene.
        head_weights : dict, optional
            Gradient-based per-head weights (for gradient_rollout).

        Returns
        -------
        edge_scores : dict
            ``{(tf_name, target_name): score}`` for every TF.
        """
        from tabpfn.grn.attention_extractor import _extract_rollout_scores_vectorized

        first_layer_key = sorted(
            attention.keys(), key=lambda x: int(x.split("_")[1])
        )[0]
        first_layer_data = attention[first_layer_key]

        if "between_items" in first_layer_data:
            num_feature_blocks = first_layer_data["between_items"].size(1)
        else:
            num_feature_blocks = len(self.tf_names) + 1

        target_idx = num_feature_blocks - 1
        head_combination = "weighted" if head_weights else "mean"

        # Build feature-to-block mapping
        from tabpfn.grn.utils import compute_feature_to_block_mapping
        f2b, _ = compute_feature_to_block_mapping(len(self.tf_names), num_feature_blocks + 1)

        if self.use_kronecker:
            scores = compute_kronecker_rollout_scores(
                attention,
                target_idx=target_idx,
                head_combination=head_combination,
                head_weights=head_weights,
                add_residual=True,
            )

            edge_scores: dict[tuple[str, str], float] = {}
            for tf_idx, tf_name in enumerate(self.tf_names):
                blocks = f2b[tf_idx]
                score = sum(scores[b].item() for b in blocks) / len(blocks)
                edge_scores[(tf_name, target_name)] = score
        else:
            aggregation = "gradient_weighted" if head_weights else "sequential_rollout"
            computer = EdgeScoreComputer(aggregation_method=aggregation)
            rollout_matrix = computer.compute(
                attention,
                use_between_features=True,
                use_between_items=True,
                head_combination=head_combination,
                head_weights=head_weights,
                add_residual=True,
                average_batch=True,
            )

            if "between_features" in first_layer_data:
                num_items = first_layer_data["between_features"].size(1)
            else:
                raise ValueError("Cannot determine num_items from attention weights")

            # Collect all unique block indices we need scores for
            all_block_indices = sorted({b for blocks in f2b.values() for b in blocks})
            edge_scores = {}

            if all_block_indices:
                tf_tensor = torch.tensor(
                    all_block_indices, device=rollout_matrix.device, dtype=torch.long
                )
                mean_scores = _extract_rollout_scores_vectorized(
                    rollout_matrix, tf_tensor, target_idx, num_items, num_feature_blocks
                )
                block_score_map = {b: mean_scores[i].item() for i, b in enumerate(all_block_indices)}

                for tf_idx, tf_name in enumerate(self.tf_names):
                    blocks = f2b[tf_idx]
                    score = sum(block_score_map[b] for b in blocks) / len(blocks)
                    edge_scores[(tf_name, target_name)] = score

        return edge_scores

    def _compute_edge_scores(self) -> dict[tuple[str, str], float]:
        """Compute edge scores from attention weights.

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) pairs to edge scores
        """
        import torch

        edge_scores = {}

        # Integrated gradients doesn't use attention weights —
        # use shared model architecture loaded once
        if self.edge_score_strategy == "integrated_gradients":
            # Get shared model arch (loaded once in fit())
            model_arch = getattr(self, '_shared_model_arch_', None)
            device = getattr(self, '_shared_device_', None)

            # Fallback: extract from first available target model
            if model_arch is None:
                for tn in self.target_genes:
                    m = self.target_models_.get(tn)
                    if m is not None and hasattr(m, "models_") and m.models_:
                        model_arch = m.models_[0]
                        device = (
                            m.devices_[0]
                            if hasattr(m, "devices_") and m.devices_
                            else torch.device(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            )
                        )
                        break

            if model_arch is None:
                for tf_name in self.tf_names:
                    for target_name in self.target_genes:
                        edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores

            if not (hasattr(self, 'X_') and hasattr(self, 'y_')):
                import warnings
                warnings.warn("Training data not available for IG.")
                for tf_name in self.tf_names:
                    for target_name in self.target_genes:
                        edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores

            X_ig = self.X_
            for target_name in self.target_genes:
                target_idx_in_y = self.target_genes.index(target_name)
                y_target = self.y_[:, target_idx_in_y]

                try:
                    target_scores = self._integrated_gradients_edge_scores(
                        model_arch, device, X_ig, y_target, target_name,
                        ig_n_folds=self.ig_n_folds,
                        baseline=self.ig_baseline,
                    )
                    edge_scores.update(target_scores)
                except Exception as e:
                    import warnings
                    import traceback
                    warnings.warn(
                        f"Failed integrated_gradients for {target_name}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    for tf_name in self.tf_names:
                        edge_scores[(tf_name, target_name)] = 0.0

            return edge_scores

        # RISE — forward-only perturbation method
        if self.edge_score_strategy == "rise":
            model_arch = getattr(self, '_shared_model_arch_', None)
            device = getattr(self, '_shared_device_', None)

            if model_arch is None:
                for tn in self.target_genes:
                    m = self.target_models_.get(tn)
                    if m is not None and hasattr(m, "models_") and m.models_:
                        model_arch = m.models_[0]
                        device = (
                            m.devices_[0]
                            if hasattr(m, "devices_") and m.devices_
                            else torch.device(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            )
                        )
                        break

            if model_arch is None or not (hasattr(self, 'X_') and hasattr(self, 'y_')):
                for tf_name in self.tf_names:
                    for target_name in self.target_genes:
                        edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores

            X_rise = self.X_
            criterion = getattr(self, '_shared_criterion_', None)
            for target_name in self.target_genes:
                target_idx_in_y = self.target_genes.index(target_name)
                y_target = self.y_[:, target_idx_in_y]

                try:
                    target_scores = self._rise_edge_scores(
                        model_arch, device, X_rise, y_target, target_name,
                        n_masks=self.rise_n_masks,
                        mask_prob=self.rise_mask_prob,
                        baseline=self.rise_baseline,
                        criterion=criterion,
                        rise_n_folds=self.rise_n_folds,
                    )
                    edge_scores.update(target_scores)
                except Exception as e:
                    import warnings
                    import traceback
                    warnings.warn(
                        f"Failed RISE for {target_name}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    for tf_name in self.tf_names:
                        edge_scores[(tf_name, target_name)] = 0.0

            return edge_scores

        # Shapley — NaN-masking based feature attribution
        if self.edge_score_strategy == "shapley":
            model_arch = getattr(self, '_shared_model_arch_', None)
            device = getattr(self, '_shared_device_', None)

            if model_arch is None:
                for tn in self.target_genes:
                    m = self.target_models_.get(tn)
                    if m is not None and hasattr(m, "models_") and m.models_:
                        model_arch = m.models_[0]
                        device = (
                            m.devices_[0]
                            if hasattr(m, "devices_") and m.devices_
                            else torch.device(
                                "cuda" if torch.cuda.is_available() else "cpu"
                            )
                        )
                        break

            if model_arch is None or not (hasattr(self, 'X_') and hasattr(self, 'y_')):
                for tf_name in self.tf_names:
                    for target_name in self.target_genes:
                        edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores

            X_shap = self.X_
            criterion = getattr(self, '_shared_criterion_', None)
            for target_name in self.target_genes:
                target_idx_in_y = self.target_genes.index(target_name)
                y_target = self.y_[:, target_idx_in_y]

                try:
                    target_scores = self._shapley_edge_scores(
                        model_arch, device, X_shap, y_target, target_name,
                        criterion=criterion,
                    )
                    edge_scores.update(target_scores)
                except Exception as e:
                    import warnings
                    import traceback
                    warnings.warn(
                        f"Failed Shapley for {target_name}: {e}\n"
                        f"{traceback.format_exc()}"
                    )
                    for tf_name in self.tf_names:
                        edge_scores[(tf_name, target_name)] = 0.0

            return edge_scores

        for target_name, attention in self.attention_weights_.items():
            # Handle sequential_rollout strategy
            if self.edge_score_strategy == "sequential_rollout":
                try:
                    target_scores = self._rollout_edge_scores(attention, target_name)
                    edge_scores.update(target_scores)
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
                    import numpy as np

                    model = self.target_models_[target_name]
                    gradient_extractor = GradientAttentionExtractor()

                    if hasattr(self, 'X_') and hasattr(self, 'y_'):
                        X_grad = self.X_
                        target_idx_in_y = self.target_genes.index(target_name)
                        y_target = self.y_[:, target_idx_in_y]
                    elif hasattr(self, 'X_sample_') and hasattr(self, 'y_sample_'):
                        X_grad = self.X_sample_
                        target_idx_in_y = self.target_genes.index(target_name)
                        y_target = self.y_sample_[:, target_idx_in_y]
                    else:
                        X_grad = np.random.randn(10, len(self.tf_names))
                        y_target = np.random.randn(10)
                        import warnings
                        warnings.warn(
                            f"Using synthetic data for gradient computation for {target_name}. "
                            "Results may be suboptimal."
                        )

                    if not isinstance(X_grad, torch.Tensor):
                        X_grad = torch.from_numpy(X_grad).float()
                    if not isinstance(y_target, torch.Tensor):
                        y_target = torch.from_numpy(y_target).float()

                    if hasattr(model, 'devices_') and model.devices_:
                        device = model.devices_[0]
                        X_grad = X_grad.to(device)
                        y_target = y_target.to(device)
                    elif torch.cuda.is_available():
                        X_grad = X_grad.cuda()
                        y_target = y_target.cuda()

                    head_weights = gradient_extractor.compute_gradient_head_weights(
                        model=model,
                        X=X_grad,
                        y_target=y_target,
                        attention_weights=attention,
                        normalization="l1",
                    )

                    target_scores = self._rollout_edge_scores(
                        attention, target_name, head_weights=head_weights
                    )
                    edge_scores.update(target_scores)

                except Exception as e:
                    import warnings
                    import traceback
                    warnings.warn(f"Failed to extract edge scores for {target_name} with gradient_rollout: {e}\n{traceback.format_exc()}")
                    # Fallback: sequential rollout (unweighted)
                    try:
                        target_scores = self._rollout_edge_scores(
                            attention, target_name
                        )
                        edge_scores.update(target_scores)
                    except Exception:
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
                # Positions 0 to (n_feat_pos-2) are TF feature blocks
                n_feat_pos = feat_attn.shape[0]
                target_pos = n_feat_pos - 1  # Last position is the target

                # Build feature-to-block mapping (TFs may share blocks)
                from tabpfn.grn.utils import compute_feature_to_block_mapping
                f2b, _ = compute_feature_to_block_mapping(len(self.tf_names), n_feat_pos)

                # Extract scores for each TF using the specified strategy
                for tf_idx, tf_name in enumerate(self.tf_names):
                    blocks = f2b[tf_idx]

                    if self.edge_score_strategy == "self_attention":
                        score = sum(feat_attn[b, b].item() for b in blocks) / len(blocks)
                    elif self.edge_score_strategy == "tf_to_target":
                        score = sum(feat_attn[b, target_pos].item() for b in blocks) / len(blocks)
                    elif self.edge_score_strategy == "target_to_tf":
                        score = sum(feat_attn[target_pos, b].item() for b in blocks) / len(blocks)
                    elif self.edge_score_strategy == "combined":
                        s_self = sum(feat_attn[b, b].item() for b in blocks) / len(blocks)
                        s_tf_targ = sum(feat_attn[b, target_pos].item() for b in blocks) / len(blocks)
                        s_targ_tf = sum(feat_attn[target_pos, b].item() for b in blocks) / len(blocks)
                        score = (s_self + s_tf_targ + s_targ_tf) / 3
                    elif self.edge_score_strategy == "combined_best":
                        s_self = sum(feat_attn[b, b].item() for b in blocks) / len(blocks)
                        s_tf_targ = sum(feat_attn[b, target_pos].item() for b in blocks) / len(blocks)
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

    def get_edge_scores(self, edge_score_strategy: str | None = None) -> dict[tuple[str, str], float]:
        """Get edge scores from fitted model.

        Parameters
        ----------
        edge_score_strategy : str, optional
            Edge score strategy to use. If None, uses the strategy from __init__.
            Options: 'self_attention', 'tf_to_target', 'target_to_tf', 'sequential_rollout',
                     'gradient_rollout', 'integrated_gradients', 'combined', 'combined_best'
            This allows computing different edge score strategies without re-fitting.

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

        # If no strategy specified or same as init, return pre-computed scores
        if edge_score_strategy is None or edge_score_strategy == self.edge_score_strategy:
            return self.edge_scores_

        # Otherwise, recompute with the new strategy
        return self._compute_edge_scores_with_strategy(edge_score_strategy)

    def _compute_edge_scores_with_strategy(self, edge_score_strategy: str) -> dict[tuple[str, str], float]:
        """Compute edge scores from attention weights using a specific strategy.

        This method allows recomputing edge scores with a different strategy
        without re-fitting the model.

        Parameters
        ----------
        edge_score_strategy : str
            Edge score strategy to use
            Options: 'self_attention', 'tf_to_target', 'target_to_tf', 'sequential_rollout',
                     'gradient_rollout', 'integrated_gradients', 'combined', 'combined_best'

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) pairs to edge scores
        """
        # Temporarily override the strategy for computation
        original_strategy = self.edge_score_strategy
        self.edge_score_strategy = edge_score_strategy

        try:
            # Use the existing _compute_edge_scores method with the new strategy
            edge_scores = self._compute_edge_scores()
            return edge_scores
        finally:
            # Restore original strategy
            self.edge_score_strategy = original_strategy

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
                edge_scores = self._rollout_edge_scores(attention, target_name)
            except Exception as e:
                import warnings
                import traceback
                warnings.warn(f"Failed to extract edge scores for {target_name}: {e}\n{traceback.format_exc()}")
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0

        # Handle gradient_rollout strategy
        elif self.edge_score_strategy == "gradient_rollout":
            if model is None:
                import warnings
                warnings.warn(f"Model not provided for {target_name}, using sequential_rollout instead")
                original_strategy = self.edge_score_strategy
                self.edge_score_strategy = "sequential_rollout"
                edge_scores = self._compute_edge_scores_from_attention(
                    attention, target_name, X, y_target, n_train
                )
                self.edge_score_strategy = original_strategy
                return edge_scores

            try:
                gradient_extractor = GradientAttentionExtractor()

                X_grad = torch.from_numpy(X).float()
                y_target_tensor = torch.from_numpy(y_target).float()

                if hasattr(model, 'devices_') and model.devices_:
                    device = model.devices_[0]
                else:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                X_grad = X_grad.to(device)
                y_target_tensor = y_target_tensor.to(device)

                head_weights = gradient_extractor.compute_gradient_head_weights(
                    model=model,
                    X=X_grad,
                    y_target=y_target_tensor,
                    attention_weights=attention,
                    normalization="l1",
                )

                edge_scores = self._rollout_edge_scores(
                    attention, target_name, head_weights=head_weights
                )

            except Exception as e:
                import warnings
                import traceback
                warnings.warn(f"Failed to extract edge scores for {target_name} with gradient_rollout: {e}\n{traceback.format_exc()}")
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0

        # Handle integrated_gradients strategy — not attention-based,
        # should be dispatched via _compute_edge_scores() instead
        elif self.edge_score_strategy == "integrated_gradients":
            import warnings
            warnings.warn(
                "integrated_gradients should not be called via "
                "_compute_edge_scores_from_attention(). Use "
                "_compute_edge_scores() instead."
            )
            if model is not None:
                edge_scores = self._integrated_gradients_edge_scores(
                    model, X, y_target, target_name, n_train=n_train,
                    ig_n_folds=self.ig_n_folds,
                    baseline=self.ig_baseline,
                )
            else:
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0

        else:
            # Handle other edge score strategies (self_attention, tf_to_target, etc.)
            # For simplicity, assign uniform scores or use the aggregation computer
            for tf_name in self.tf_names:
                edge_scores[(tf_name, target_name)] = 0.0

        return edge_scores

    def _integrated_gradients_edge_scores(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_target: np.ndarray,
        target_name: str,
        n_train: int | None = None,
        n_steps: int = 50,
        ig_n_folds: int = 1,
        baseline: str = "zero",
    ) -> dict[tuple[str, str], float]:
        """Compute edge scores using Integrated Gradients (batched on GPU).

        Uses TabPFN's batch dimension to run all interpolation steps (and
        optionally all CV folds) in a single forward pass, achieving up to
        ``n_steps * n_folds`` x speedup over the sequential version.

        Parameters
        ----------
        model_arch : torch.nn.Module
            The underlying TabPFN architecture (e.g., PerFeatureTransformer).
            Shared across all targets — loaded once.
        device : torch.device
            Device to run computation on.
        X : np.ndarray
            Input data of shape ``(n_samples, n_TFs)``.
        y_target : np.ndarray
            Target gene expression of shape ``(n_samples,)``.
        target_name : str
            Name of the target gene.
        n_train : int, optional
            Number of training samples for the train/test split. Only used
            when ``ig_n_folds=1``. If ``None``, uses 80% of ``X``.
        n_steps : int, default=50
            Number of interpolation steps along the path from baseline to input.
        ig_n_folds : int, default=1
            Number of cross-validation folds for IG. When >1, rotates the
            train/test split across folds and averages the per-TF scores.
        baseline : str, default="zero"
            Baseline choice: ``"zero"`` or ``"mean"`` (per-fold training mean).

        Returns
        -------
        edge_scores : dict
            ``{(tf_name, target_name): score}`` for every TF.
        """
        import torch

        edge_scores: dict[tuple[str, str], float] = {}

        n_samples, n_features = X.shape

        # Build list of (X_fold, y_fold_train, n_train_fold) for each fold
        fold_specs: list[tuple[np.ndarray, np.ndarray, int]] = []

        if ig_n_folds > 1:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=ig_n_folds, shuffle=True,
                random_state=self.random_state,
            )
            for train_idx, test_idx in kfold.split(X):
                X_fold = np.concatenate([X[train_idx], X[test_idx]], axis=0)
                y_fold_train = y_target[train_idx]
                fold_specs.append((X_fold, y_fold_train, len(train_idx)))
        else:
            if n_train is None:
                n_train = max(1, int(n_samples * 0.8))
            if n_samples - n_train < 1:
                import warnings
                warnings.warn(
                    f"Not enough samples for IG split (n={n_samples})."
                )
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores
            fold_specs.append((X, y_target[:n_train], n_train))

        # Group folds by n_train (single_eval_pos must be same within a batch)
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for fold_idx, (_, _, nt) in enumerate(fold_specs):
            groups[nt].append(fold_idx)

        accumulated_scores = np.zeros(n_features)

        for group_n_train, fold_indices in groups.items():
            group_scores = self._ig_batched(
                model_arch, device, fold_specs, fold_indices,
                group_n_train, n_steps, baseline,
            )
            # group_scores shape: (n_folds_in_group, n_features)
            accumulated_scores += group_scores.sum(axis=0)

        tf_scores = accumulated_scores / len(fold_specs)

        for tf_idx, tf_name in enumerate(self.tf_names):
            edge_scores[(tf_name, target_name)] = float(tf_scores[tf_idx])

        return edge_scores

    def _ig_batched(
        self,
        model_arch: torch.nn.Module,
        device: torch.device,
        fold_specs: list[tuple[np.ndarray, np.ndarray, int]],
        fold_indices: list[int],
        n_train: int,
        n_steps: int,
        baseline_type: str,
        max_batch: int | None = None,
    ) -> np.ndarray:
        """Run batched IG for folds that share the same ``n_train``.

        Stacks all ``n_steps * len(fold_indices)`` interpolation points along
        TabPFN's batch dimension and runs a single forward + backward pass,
        exploiting GPU parallelism.

        Parameters
        ----------
        model_arch : torch.nn.Module
            The underlying TabPFN architecture.
        device : torch.device
            Computation device.
        fold_specs : list of (X_fold, y_fold_train, n_train)
            All fold specifications (only those in ``fold_indices`` are used).
        fold_indices : list of int
            Indices into ``fold_specs`` for folds in this group.
        n_train : int
            Number of training samples (same for all folds in this group).
        n_steps : int
            Number of IG interpolation steps.
        baseline_type : str
            ``"zero"`` or ``"mean"``.
        max_batch : int or None
            Maximum batch size to avoid CUDA OOM. If ``None``, automatically
            estimated from available GPU memory. Larger batches are chunked.

        Returns
        -------
        fold_scores : np.ndarray
            Shape ``(len(fold_indices), n_features)`` — per-TF IG scores per fold.
        """
        import torch

        n_folds = len(fold_indices)
        seq_len = fold_specs[fold_indices[0]][0].shape[0]
        n_features = fold_specs[fold_indices[0]][0].shape[1]
        total_batch = n_folds * n_steps
        # Avoid alpha=0 which creates all-zero inputs that break gradient
        # flow through TabPFN's feature normalization layers.
        alphas = torch.linspace(1e-6, 1, n_steps, device=device)

        # Auto-detect safe batch size from available GPU memory.
        # Attention memory scales as O(batch * seq_len^2 * n_heads * n_layers).
        # Use a conservative heuristic: ~2 MB per batch item for seq_len~100.
        if max_batch is None:
            if device.type == "cuda":
                try:
                    free_mem = torch.cuda.mem_get_info(device)[0]
                    # Reserve 512MB for overhead; ~2MB per batch item
                    max_batch = max(1, int((free_mem - 512 * 1024**2) / (2 * 1024**2)))
                    max_batch = min(max_batch, 200)  # cap for safety
                except Exception:
                    max_batch = 32
            else:
                max_batch = total_batch  # CPU: no limit

        # Pre-compute per-fold tensors
        X_folds = []      # (n_folds, seq_len, n_features)
        baselines = []     # (n_folds, seq_len, n_features)
        y_trains = []      # (n_folds, n_train)
        for fi in fold_indices:
            X_f, y_f_train, _ = fold_specs[fi]
            X_t = torch.from_numpy(X_f).float().to(device)
            X_folds.append(X_t)

            if baseline_type == "mean":
                train_mean = X_t[:n_train].mean(dim=0, keepdim=True)
                baselines.append(train_mean.expand_as(X_t))
            else:
                baselines.append(torch.zeros_like(X_t))

            y_trains.append(
                torch.from_numpy(y_f_train).float().to(device)
            )

        # Stack into tensors: (n_folds, seq_len, n_features)
        X_stack = torch.stack(X_folds)        # (F, S, D)
        B_stack = torch.stack(baselines)      # (F, S, D)
        delta = X_stack - B_stack             # (F, S, D)

        model_arch.eval()
        prev_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        try:
            # Accumulate gradients across chunks
            # We need: mean_grad per fold = sum(grad_at_alpha) / n_steps
            # IG per fold = delta * mean_grad
            # score per fold = mean(|IG|, dim=samples)
            accumulated_grads = torch.zeros(
                n_folds, seq_len, n_features, device=device
            )

            # Process in chunks, halving batch on OOM
            steps_per_chunk = min(n_steps, max(1, max_batch // n_folds))

            s_start = 0
            while s_start < n_steps:
                s_end = min(s_start + steps_per_chunk, n_steps)
                chunk_steps = s_end - s_start
                chunk_batch = n_folds * chunk_steps

                alpha_expanded = alphas[s_start:s_end].reshape(
                    1, 1, chunk_steps, 1
                )
                delta_exp = delta.unsqueeze(2)
                B_exp = B_stack.unsqueeze(2)

                interp = B_exp + alpha_expanded * delta_exp
                interp = interp.permute(1, 0, 2, 3).reshape(
                    seq_len, chunk_batch, n_features
                )

                X_input = interp.clone().detach().requires_grad_(True)

                y_parts = []
                for fi_local in range(n_folds):
                    y_f = y_trains[fi_local].unsqueeze(1).unsqueeze(2)
                    y_parts.append(y_f.expand(n_train, chunk_steps, 1))
                y_input = torch.cat(y_parts, dim=1)

                try:
                    output = model_arch.forward(
                        x={"main": X_input},
                        y={"main": y_input},
                        only_return_standard_out=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    # Halve chunk size and retry this chunk
                    torch.cuda.empty_cache()
                    steps_per_chunk = max(1, steps_per_chunk // 2)
                    continue

                if isinstance(output, dict):
                    y_pred = output.get(
                        "standard", next(iter(output.values()))
                    )
                elif isinstance(output, (list, tuple)):
                    y_pred = output[0]
                else:
                    y_pred = output

                if y_pred is None or y_pred.numel() == 0:
                    s_start = s_end
                    continue
                if not y_pred.requires_grad:
                    s_start = s_end
                    continue

                pred_sum = y_pred.sum()
                try:
                    pred_sum.backward()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    steps_per_chunk = max(1, steps_per_chunk // 2)
                    continue

                if X_input.grad is not None:
                    grad = X_input.grad.detach()
                    grad = grad.reshape(
                        seq_len, n_folds, chunk_steps, n_features
                    )
                    accumulated_grads += grad.sum(dim=2).permute(1, 0, 2)

                s_start = s_end

            # IG = delta * mean_grad_over_steps
            mean_grads = accumulated_grads / n_steps  # (F, S, D)
            ig = delta * mean_grads                    # (F, S, D)
            # Per-fold score = mean(|IG|) over samples
            fold_scores = ig.abs().mean(dim=1).cpu().numpy()  # (F, D)

        except Exception as e:
            import traceback
            import warnings
            warnings.warn(
                f"Batched IG failed: {e}\n{traceback.format_exc()}"
            )
            fold_scores = np.zeros((n_folds, n_features))
        finally:
            torch.set_grad_enabled(prev_grad)

        return fold_scores

    # ------------------------------------------------------------------
    # RISE (Randomized Input Sampling for Explanation)
    # ------------------------------------------------------------------

    def _rise_edge_scores(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_target: np.ndarray,
        target_name: str,
        n_train: int | None = None,
        n_masks: int = 500,
        mask_prob: float = 0.5,
        baseline: str = "zero",
        criterion=None,
        rise_n_folds: int = 1,
    ) -> dict[tuple[str, str], float]:
        """Compute edge scores using RISE with optional cross-validation.

        Generates random binary masks over input features, runs forward
        passes with masked inputs, and attributes each TF based on the
        correlation between its presence and prediction accuracy.

        Parameters
        ----------
        model_arch : torch.nn.Module
            The underlying TabPFN architecture.
        device : torch.device
            Device for computation.
        X : np.ndarray
            Input data of shape ``(n_samples, n_TFs)``.
        y_target : np.ndarray
            Target gene expression of shape ``(n_samples,)``.
        target_name : str
            Name of the target gene.
        n_train : int, optional
            Number of training samples. If ``None``, uses 80%.
        n_masks : int, default=500
            Number of random binary masks.
        mask_prob : float, default=0.5
            Probability of keeping each feature per mask.
        baseline : str, default="zero"
            Fill value for masked features: ``"zero"`` or ``"mean"``.
        criterion : FullSupportBarDistribution, optional
            Bar distribution for converting logits to mean predictions.
        rise_n_folds : int, default=1
            Number of CV folds. When >1, rotates train/test splits and
            averages per-TF scores across folds.

        Returns
        -------
        edge_scores : dict
            ``{(tf_name, target_name): score}`` for every TF.
        """
        import torch

        n_samples, n_features = X.shape

        # Build fold specifications (same pattern as IG)
        fold_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

        if rise_n_folds > 1:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=rise_n_folds, shuffle=True,
                random_state=self.random_state,
            )
            for train_idx, test_idx in kfold.split(X):
                X_fold = np.concatenate([X[train_idx], X[test_idx]], axis=0)
                y_train = y_target[train_idx]
                y_test = y_target[test_idx]
                fold_specs.append((X_fold, y_train, y_test, len(train_idx)))
        else:
            if n_train is None:
                n_train = max(1, int(n_samples * 0.8))
            if n_samples - n_train < 1:
                import warnings
                warnings.warn(f"Not enough samples for RISE split (n={n_samples}).")
                edge_scores = {}
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores
            fold_specs.append((
                X, y_target[:n_train], y_target[n_train:], n_train
            ))

        # Run RISE on each fold, accumulate scores
        accumulated_scores = np.zeros(n_features)
        for X_fold, y_train, y_test, nt in fold_specs:
            fold_scores = self._rise_single_fold(
                model_arch, device, X_fold, y_train, y_test, nt,
                n_masks=n_masks, mask_prob=mask_prob,
                baseline=baseline, criterion=criterion,
            )
            accumulated_scores += fold_scores

        tf_scores = accumulated_scores / len(fold_specs)

        edge_scores: dict[tuple[str, str], float] = {}
        for tf_idx, tf_name in enumerate(self.tf_names):
            edge_scores[(tf_name, target_name)] = float(tf_scores[tf_idx])

        return edge_scores

    def _rise_single_fold(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_train: int,
        n_masks: int,
        mask_prob: float,
        baseline: str,
        criterion=None,
    ) -> np.ndarray:
        """Run RISE for a single train/test split.

        Returns
        -------
        scores : np.ndarray
            Shape ``(n_features,)`` — per-TF RISE importance scores.
        """
        import torch

        n_samples, n_features = X.shape
        n_test = n_samples - n_train

        X_t = torch.from_numpy(X).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        # Compute baseline fill values
        if baseline == "mean":
            fill_values = X_t[:n_train].mean(dim=0)
        else:
            fill_values = torch.zeros(n_features, device=device)

        # Generate random binary masks
        rng = np.random.RandomState(self.random_state)
        masks_np = (rng.rand(n_masks, n_features) < mask_prob).astype(np.float32)
        masks = torch.from_numpy(masks_np).to(device)

        # Auto-detect max batch size
        if device.type == "cuda":
            try:
                free_mem = torch.cuda.mem_get_info(device)[0]
                max_batch = max(1, int((free_mem - 512 * 1024**2) / (2 * 1024**2)))
                max_batch = min(max_batch, 500)
            except Exception:
                max_batch = 50
        else:
            max_batch = n_masks

        model_arch.eval()
        weighted_sum = torch.zeros(n_features, device=device)
        mask_count = torch.zeros(n_features, device=device)
        total_metric_sum = torch.zeros(n_features, device=device)

        m_start = 0
        chunk_size = min(n_masks, max_batch)
        while m_start < n_masks:
            m_end = min(m_start + chunk_size, n_masks)
            chunk_n = m_end - m_start
            chunk_masks = masks[m_start:m_end]

            X_expanded = X_t.unsqueeze(1).expand(n_samples, chunk_n, n_features)
            mask_expanded = chunk_masks.unsqueeze(0).expand(n_samples, chunk_n, n_features)
            fill_expanded = fill_values.unsqueeze(0).unsqueeze(0).expand(
                n_samples, chunk_n, n_features
            )
            X_masked = X_expanded * mask_expanded + fill_expanded * (1 - mask_expanded)

            y_input = y_train_t.unsqueeze(1).unsqueeze(2).expand(n_train, chunk_n, 1)

            try:
                with torch.no_grad():
                    output = model_arch.forward(
                        x={"main": X_masked},
                        y={"main": y_input},
                        only_return_standard_out=True,
                    )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                continue

            if isinstance(output, dict):
                y_pred = output.get("standard", next(iter(output.values())))
            elif isinstance(output, (list, tuple)):
                y_pred = output[0]
            else:
                y_pred = output

            if y_pred is None or y_pred.numel() == 0:
                m_start = m_end
                continue

            # Convert logits to mean predictions
            if criterion is not None:
                mean_preds = criterion.mean(y_pred)  # (n_test, chunk_n)
            else:
                mean_preds = y_pred.mean(dim=-1)
            # Negative MSE per mask
            y_test_expanded = y_test_t.unsqueeze(1).expand(n_test, chunk_n)
            pred_metric = -((mean_preds - y_test_expanded) ** 2).mean(dim=0)

            weighted_sum += (pred_metric.unsqueeze(1) * chunk_masks).sum(dim=0)
            mask_count += chunk_masks.sum(dim=0)
            total_metric_sum += pred_metric.sum().expand(n_features)

            m_start = m_end

        # RISE score: avg metric when present - avg metric when absent
        mask_count = mask_count.clamp(min=1)
        absent_count = (n_masks - mask_count).clamp(min=1)
        present_avg = weighted_sum / mask_count
        absent_sum = total_metric_sum - weighted_sum
        absent_avg = absent_sum / absent_count
        rise_scores = (present_avg - absent_avg).abs()

        return rise_scores.cpu().numpy()

    # ------------------------------------------------------------------
    # Shapley value edge scores (NaN-masking)
    # ------------------------------------------------------------------

    def _shapley_edge_scores(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_target: np.ndarray,
        target_name: str,
        n_train: int | None = None,
        criterion=None,
    ) -> dict[tuple[str, str], float]:
        """Compute edge scores using Shapley values with optional CV.

        Uses NaN-masking to simulate feature removal, leveraging TabPFN's
        built-in NaN handling (NanHandlingEncoderStep) so the model sees
        features as genuinely absent rather than zeroed.

        For n_TFs ≤ shapley_exact_threshold: exact Shapley (all 2^n coalitions).
        Otherwise: permutation-based approximation.

        Parameters
        ----------
        model_arch : torch.nn.Module
            Pre-loaded TabPFN architecture.
        device : torch.device
            Computation device.
        X : np.ndarray
            TF expression matrix (n_samples, n_features).
        y_target : np.ndarray
            Target gene expression (n_samples,).
        target_name : str
            Name of target gene.
        n_train : int, optional
            Number of training samples for single-fold.
        criterion : FullSupportBarDistribution, optional
            Bar distribution for converting logits to mean predictions.

        Returns
        -------
        dict of (str, str) → float
            Edge scores for (TF, target) pairs.
        """
        import torch

        n_samples, n_features = X.shape
        n_folds = self.shapley_n_folds

        # Build fold specs (same pattern as IG/RISE)
        fold_specs: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []

        if n_folds > 1:
            from sklearn.model_selection import KFold
            kfold = KFold(
                n_splits=n_folds, shuffle=True,
                random_state=self.random_state,
            )
            for train_idx, test_idx in kfold.split(X):
                X_fold = np.concatenate([X[train_idx], X[test_idx]], axis=0)
                y_train = y_target[train_idx]
                y_test = y_target[test_idx]
                fold_specs.append((X_fold, y_train, y_test, len(train_idx)))
        else:
            if n_train is None:
                n_train = max(1, int(n_samples * 0.8))
            if n_samples - n_train < 1:
                import warnings
                warnings.warn(
                    f"Not enough samples for Shapley split (n={n_samples})."
                )
                edge_scores = {}
                for tf_name in self.tf_names:
                    edge_scores[(tf_name, target_name)] = 0.0
                return edge_scores
            fold_specs.append((
                X, y_target[:n_train], y_target[n_train:], n_train
            ))

        accumulated_scores = np.zeros(n_features)
        for X_fold, y_train, y_test, nt in fold_specs:
            fold_scores = self._shapley_single_fold(
                model_arch, device, X_fold, y_train, y_test, nt,
                criterion=criterion,
            )
            accumulated_scores += fold_scores

        tf_scores = accumulated_scores / len(fold_specs)

        edge_scores: dict[tuple[str, str], float] = {}
        for tf_idx, tf_name in enumerate(self.tf_names):
            edge_scores[(tf_name, target_name)] = float(tf_scores[tf_idx])

        return edge_scores

    def _shapley_single_fold(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_train: int,
        criterion=None,
    ) -> np.ndarray:
        """Compute Shapley values for a single train/test split.

        Returns
        -------
        scores : np.ndarray
            Shape ``(n_features,)`` — per-TF Shapley importance scores.
        """
        n_samples, n_features = X.shape
        method = self.shapley_method

        if method.startswith("kernelshap"):
            return self._shapley_kernelshap(
                model_arch, device, X, y_train, y_test, n_train,
                criterion=criterion, mode=method,
            )

        # "auto": exact if small, permutation otherwise
        if n_features <= self.shapley_exact_threshold:
            return self._shapley_exact(
                model_arch, device, X, y_train, y_test, n_train,
                criterion=criterion,
            )
        else:
            return self._shapley_permutation(
                model_arch, device, X, y_train, y_test, n_train,
                criterion=criterion,
            )

    def _shapley_eval_coalition(
        self,
        model_arch,
        device,
        X_t: "torch.Tensor",
        y_train_t: "torch.Tensor",
        y_test_t: "torch.Tensor",
        n_train: int,
        coalition_masks: "torch.Tensor",
        criterion=None,
    ) -> "torch.Tensor":
        """Evaluate prediction quality for a batch of coalitions.

        Parameters
        ----------
        X_t : Tensor (n_samples, n_features)
        y_train_t : Tensor (n_train,)
        y_test_t : Tensor (n_test,)
        coalition_masks : Tensor (n_coalitions, n_features)
            Binary: 1 = feature present, 0 = feature NaN-masked.

        Returns
        -------
        metrics : Tensor (n_coalitions,)
            Negative MSE per coalition (higher = better).
        """
        import torch

        n_samples, n_features = X_t.shape
        n_test = n_samples - n_train
        n_coalitions = coalition_masks.shape[0]

        # Expand X and apply NaN masking
        # X_expanded: (n_samples, n_coalitions, n_features)
        X_expanded = X_t.unsqueeze(1).expand(n_samples, n_coalitions, n_features).clone()
        # Where mask is 0, set to NaN
        nan_mask = (coalition_masks == 0).unsqueeze(0).expand(
            n_samples, n_coalitions, n_features
        )
        X_expanded[nan_mask] = float('nan')

        y_input = y_train_t.unsqueeze(1).unsqueeze(2).expand(n_train, n_coalitions, 1)

        with torch.no_grad():
            output = model_arch.forward(
                x={"main": X_expanded},
                y={"main": y_input},
                only_return_standard_out=True,
            )

        if isinstance(output, dict):
            y_pred = output.get("standard", next(iter(output.values())))
        elif isinstance(output, (list, tuple)):
            y_pred = output[0]
        else:
            y_pred = output

        if y_pred is None or y_pred.numel() == 0:
            return torch.zeros(n_coalitions, device=device)

        # Convert logits to mean predictions
        if criterion is not None:
            mean_preds = criterion.mean(y_pred)  # (n_test, n_coalitions)
        else:
            mean_preds = y_pred.mean(dim=-1)

        # Negative MSE per coalition
        y_test_expanded = y_test_t.unsqueeze(1).expand(n_test, n_coalitions)
        metrics = -((mean_preds - y_test_expanded) ** 2).mean(dim=0)  # (n_coalitions,)
        return metrics

    def _shapley_exact(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_train: int,
        criterion=None,
    ) -> np.ndarray:
        """Exact Shapley values by enumerating all 2^n coalitions."""
        import torch
        from itertools import combinations as itertools_combinations

        n_samples, n_features = X.shape
        n_coalitions = 2 ** n_features

        X_t = torch.from_numpy(X).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        # Build all coalition masks: (2^n, n_features)
        all_masks = torch.zeros(n_coalitions, n_features, device=device)
        for idx in range(n_coalitions):
            for j in range(n_features):
                if idx & (1 << j):
                    all_masks[idx, j] = 1.0

        # Evaluate all coalitions in chunks
        model_arch.eval()
        all_metrics = torch.zeros(n_coalitions, device=device)

        # Auto-detect max batch
        if device.type == "cuda":
            try:
                free_mem = torch.cuda.mem_get_info(device)[0]
                max_batch = max(1, int((free_mem - 512 * 1024**2) / (2 * 1024**2)))
                max_batch = min(max_batch, 500)
            except Exception:
                max_batch = 50
        else:
            max_batch = n_coalitions

        chunk_size = min(n_coalitions, max_batch)
        c_start = 0
        while c_start < n_coalitions:
            c_end = min(c_start + chunk_size, n_coalitions)
            try:
                chunk_metrics = self._shapley_eval_coalition(
                    model_arch, device, X_t, y_train_t, y_test_t,
                    n_train, all_masks[c_start:c_end], criterion,
                )
                all_metrics[c_start:c_end] = chunk_metrics
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                continue
            c_start = c_end

        # Compute exact Shapley values from coalition metrics
        # φ_j = Σ_{S ⊆ N\{j}} [|S|!(n-|S|-1)!/n!] [v(S∪{j}) - v(S)]
        import math
        shapley_values = torch.zeros(n_features, device=device)
        n = n_features
        factorial_n = math.factorial(n)

        for j in range(n_features):
            for idx in range(n_coalitions):
                # Check if j is NOT in coalition idx
                if idx & (1 << j):
                    continue
                # S = coalition idx (without j)
                s_size = bin(idx).count('1')
                # v(S ∪ {j}) - v(S)
                idx_with_j = idx | (1 << j)
                marginal = all_metrics[idx_with_j] - all_metrics[idx]
                weight = math.factorial(s_size) * math.factorial(n - s_size - 1) / factorial_n
                shapley_values[j] += weight * marginal

        return shapley_values.abs().cpu().numpy()

    def _shapley_permutation(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_train: int,
        criterion=None,
    ) -> np.ndarray:
        """Approximate Shapley values via permutation sampling."""
        import torch

        n_samples, n_features = X.shape
        n_perms = self.shapley_n_permutations

        X_t = torch.from_numpy(X).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device)

        rng = np.random.RandomState(self.random_state)
        shapley_values = torch.zeros(n_features, device=device)

        model_arch.eval()

        # Auto-detect max batch
        if device.type == "cuda":
            try:
                free_mem = torch.cuda.mem_get_info(device)[0]
                max_batch = max(1, int((free_mem - 512 * 1024**2) / (2 * 1024**2)))
                max_batch = min(max_batch, 500)
            except Exception:
                max_batch = 50
        else:
            max_batch = n_features + 1

        for perm_idx in range(n_perms):
            perm = rng.permutation(n_features)

            # Build n+1 coalition masks for this permutation:
            # empty → {perm[0]} → {perm[0],perm[1]} → ... → all features
            masks = torch.zeros(n_features + 1, n_features, device=device)
            for k in range(n_features):
                masks[k + 1] = masks[k].clone()
                masks[k + 1, perm[k]] = 1.0

            # Evaluate all n+1 coalitions
            chunk_size = min(n_features + 1, max_batch)
            all_metrics = torch.zeros(n_features + 1, device=device)
            c_start = 0
            while c_start < n_features + 1:
                c_end = min(c_start + chunk_size, n_features + 1)
                try:
                    chunk_metrics = self._shapley_eval_coalition(
                        model_arch, device, X_t, y_train_t, y_test_t,
                        n_train, masks[c_start:c_end], criterion,
                    )
                    all_metrics[c_start:c_end] = chunk_metrics
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk_size = max(1, chunk_size // 2)
                    continue
                c_start = c_end

            # Marginal contribution of each feature in this permutation
            for k in range(n_features):
                feature_j = perm[k]
                marginal = all_metrics[k + 1] - all_metrics[k]
                shapley_values[feature_j] += marginal

        shapley_values /= n_perms
        return shapley_values.abs().cpu().numpy()

    def _shapley_kernelshap(
        self,
        model_arch,
        device,
        X: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        n_train: int,
        criterion=None,
        mode: str = "kernelshap_full",
    ) -> np.ndarray:
        """Compute Shapley values using the shap.KernelExplainer package.

        Three modes adapted for TabPFN's in-context learning:

        - ``kernelshap_test``: Mask features in test samples only.
          Measures direct feature effect on predictions.
        - ``kernelshap_train``: Mask features in train samples only.
          Measures in-context learning contribution of each TF.
        - ``kernelshap_full``: Mask features in both train and test.
          Measures total TF contribution (learning + prediction).

        Returns
        -------
        scores : np.ndarray
            Shape ``(n_features,)`` — per-TF importance (mean |SHAP|).
        """
        import shap
        import torch

        n_samples, n_features = X.shape
        n_test = n_samples - n_train

        X_train_np = X[:n_train]
        X_test_np = X[n_train:]

        X_t = torch.from_numpy(X).float().to(device)
        y_train_t = torch.from_numpy(y_train).float().to(device)

        model_arch.eval()

        def _forward_predict(X_input_np):
            """Run TabPFN forward and return mean predictions for test samples."""
            X_in = torch.from_numpy(X_input_np.astype(np.float32)).to(device)
            seq_len = X_in.shape[0]
            batch = 1
            X_3d = X_in.unsqueeze(1)  # (seq_len, 1, n_features)
            y_in = y_train_t.unsqueeze(1).unsqueeze(2)  # (n_train, 1, 1)

            with torch.no_grad():
                output = model_arch.forward(
                    x={"main": X_3d},
                    y={"main": y_in},
                    only_return_standard_out=True,
                )

            if isinstance(output, dict):
                y_pred = output.get("standard", next(iter(output.values())))
            elif isinstance(output, (list, tuple)):
                y_pred = output[0]
            else:
                y_pred = output

            if y_pred is None or y_pred.numel() == 0:
                return np.zeros(seq_len - n_train)

            if criterion is not None:
                preds = criterion.mean(y_pred).squeeze(-1)  # (n_test,) or (n_test, 1)
            else:
                preds = y_pred.mean(dim=-1).squeeze(-1)

            return preds.cpu().numpy().flatten()

        if mode == "kernelshap_test":
            # Mask test features only; train data is fixed
            def predict_fn(X_test_masked):
                """X_test_masked: (n_samples_shap, n_features)."""
                results = []
                for i in range(X_test_masked.shape[0]):
                    X_combined = np.vstack([X_train_np, X_test_masked[i:i+1]])
                    preds = _forward_predict(X_combined)
                    results.append(preds[0] if preds.size > 0 else 0.0)
                return np.array(results)

            background = shap.sample(X_train_np, min(50, n_train))
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_test_np, nsamples=128, silent=True)
            # shap_values: (n_test, n_features)
            return np.mean(np.abs(shap_values), axis=0)

        elif mode == "kernelshap_train":
            # Mask train features only; test data is fixed
            # For each test sample, compute SHAP over training features
            all_shap = np.zeros(n_features)

            def make_train_predict(x_test_row):
                def predict_fn(X_train_masked):
                    results = []
                    for i in range(X_train_masked.shape[0]):
                        X_combined = np.vstack([X_train_masked[i:i+1].repeat(n_train, axis=0), x_test_row.reshape(1, -1)])
                        # Need to reconstruct: masked train (n_train rows) + fixed test (1 row)
                        # X_train_masked[i] is one background-substituted version of a single train row
                        # But KernelSHAP gives us full masked train sets? No — it masks features.
                        # We need: for each mask, apply it to ALL train rows
                        X_tr_masked = X_train_np.copy()
                        # Replace masked features with the background values
                        mask = ~np.isnan(X_train_masked[i]) if np.any(np.isnan(X_train_masked[i])) else np.ones(n_features, dtype=bool)
                        X_combined_full = np.vstack([X_tr_masked, x_test_row.reshape(1, -1)])
                        preds = _forward_predict(X_combined_full)
                        results.append(preds[0] if preds.size > 0 else 0.0)
                    return np.array(results)
                return predict_fn

            # Train-only is complex with KernelExplainer's API.
            # Simpler approach: use _shapley_eval_coalition with train-only NaN-masking
            X_t_full = torch.from_numpy(X).float().to(device)
            y_test_t = torch.from_numpy(y_test).float().to(device)

            # Use exact or sampled coalitions, but only NaN-mask training rows
            if n_features <= self.shapley_exact_threshold:
                n_coalitions = 2 ** n_features
                all_masks = torch.zeros(n_coalitions, n_features, device=device)
                for idx in range(n_coalitions):
                    for j in range(n_features):
                        if idx & (1 << j):
                            all_masks[idx, j] = 1.0
            else:
                # Sample coalitions
                rng = np.random.RandomState(self.random_state)
                n_coalitions = min(2 * n_features + 2048, 2 ** n_features)
                masks_np = (rng.rand(n_coalitions, n_features) > 0.5).astype(np.float32)
                all_masks = torch.from_numpy(masks_np).to(device)

            # Evaluate coalitions with train-only NaN masking
            all_metrics = torch.zeros(n_coalitions, device=device)
            max_batch = 50
            c_start = 0
            chunk_size = min(n_coalitions, max_batch)
            while c_start < n_coalitions:
                c_end = min(c_start + chunk_size, n_coalitions)
                chunk_masks = all_masks[c_start:c_end]
                chunk_n = c_end - c_start

                # NaN-mask only train rows; keep test rows intact
                X_expanded = X_t_full.unsqueeze(1).expand(
                    n_samples, chunk_n, n_features
                ).clone()
                nan_mask_train = (chunk_masks == 0).unsqueeze(0).expand(
                    n_train, chunk_n, n_features
                )
                X_expanded[:n_train][nan_mask_train] = float('nan')
                # Test rows remain unmasked

                y_input = y_train_t.unsqueeze(1).unsqueeze(2).expand(n_train, chunk_n, 1)

                try:
                    with torch.no_grad():
                        output = model_arch.forward(
                            x={"main": X_expanded},
                            y={"main": y_input},
                            only_return_standard_out=True,
                        )
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk_size = max(1, chunk_size // 2)
                    continue

                if isinstance(output, dict):
                    y_pred = output.get("standard", next(iter(output.values())))
                elif isinstance(output, (list, tuple)):
                    y_pred = output[0]
                else:
                    y_pred = output

                if y_pred is not None and y_pred.numel() > 0:
                    if criterion is not None:
                        mean_preds = criterion.mean(y_pred)
                    else:
                        mean_preds = y_pred.mean(dim=-1)
                    y_test_exp = y_test_t.unsqueeze(1).expand(n_test, chunk_n)
                    metrics = -((mean_preds - y_test_exp) ** 2).mean(dim=0)
                    all_metrics[c_start:c_end] = metrics

                c_start = c_end

            # Solve weighted linear regression for Shapley values
            Z = all_masks.cpu().numpy()  # (n_coalitions, n_features)
            v = all_metrics.cpu().numpy()  # (n_coalitions,)
            shap_values = self._solve_kernelshap_wls(Z, v, n_features)
            return np.abs(shap_values)

        else:  # kernelshap_full
            # Mask features in both train and test (same as current NaN-masking)
            # Reuse _shapley_eval_coalition with all coalitions
            X_t_full = torch.from_numpy(X).float().to(device)
            y_test_t = torch.from_numpy(y_test).float().to(device)

            if n_features <= self.shapley_exact_threshold:
                n_coalitions = 2 ** n_features
                all_masks = torch.zeros(n_coalitions, n_features, device=device)
                for idx in range(n_coalitions):
                    for j in range(n_features):
                        if idx & (1 << j):
                            all_masks[idx, j] = 1.0
            else:
                rng = np.random.RandomState(self.random_state)
                n_coalitions = min(2 * n_features + 2048, 2 ** n_features)
                masks_np = (rng.rand(n_coalitions, n_features) > 0.5).astype(np.float32)
                all_masks = torch.from_numpy(masks_np).to(device)

            all_metrics = torch.zeros(n_coalitions, device=device)
            max_batch = 50
            c_start = 0
            chunk_size = min(n_coalitions, max_batch)
            while c_start < n_coalitions:
                c_end = min(c_start + chunk_size, n_coalitions)
                try:
                    chunk_metrics = self._shapley_eval_coalition(
                        model_arch, device, X_t_full, y_train_t, y_test_t,
                        n_train, all_masks[c_start:c_end], criterion,
                    )
                    all_metrics[c_start:c_end] = chunk_metrics
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk_size = max(1, chunk_size // 2)
                    continue
                c_start = c_end

            Z = all_masks.cpu().numpy()
            v = all_metrics.cpu().numpy()
            shap_values = self._solve_kernelshap_wls(Z, v, n_features)
            return np.abs(shap_values)

    @staticmethod
    def _solve_kernelshap_wls(
        Z: np.ndarray, v: np.ndarray, n_features: int,
    ) -> np.ndarray:
        """Solve KernelSHAP weighted least squares.

        Parameters
        ----------
        Z : np.ndarray (n_coalitions, n_features)
            Binary coalition masks.
        v : np.ndarray (n_coalitions,)
            Value function for each coalition.
        n_features : int
            Number of features.

        Returns
        -------
        shap_values : np.ndarray (n_features,)
        """
        from scipy.special import comb

        n = n_features
        n_coalitions = Z.shape[0]

        # Compute SHAP kernel weights
        weights = np.zeros(n_coalitions)
        for i in range(n_coalitions):
            s = int(Z[i].sum())
            if s == 0 or s == n:
                weights[i] = 1e6  # Large weight for empty/full coalitions
            else:
                denom = comb(n, s, exact=True) * s * (n - s)
                weights[i] = (n - 1) / denom if denom > 0 else 1e6

        # Weighted least squares: minimize Σ w_i (v_i - φ_0 - Σ φ_j z_ij)²
        # Add intercept column
        Z_aug = np.column_stack([np.ones(n_coalitions), Z])
        W = np.diag(weights)

        # Solve: (Z^T W Z) β = Z^T W v
        ZtW = Z_aug.T @ W
        try:
            beta = np.linalg.solve(ZtW @ Z_aug, ZtW @ v)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(Z_aug * np.sqrt(weights)[:, None],
                                    v * np.sqrt(weights), rcond=None)[0]

        # β[0] = intercept (φ_0), β[1:] = Shapley values
        return beta[1:]

    def cleanup_model(self, target_name: str | None = None) -> None:
        """Delete the fitted TabPFN model to free GPU memory.

        This method removes the heavy TabPFNRegressor model while preserving
        the attention weights needed for edge score computation. This is essential
        when fitting multiple targets to avoid OOM errors.

        Parameters
        ----------
        target_name : str, optional
            Name of the target model to delete. If None, deletes all models.
            Use this when fitting targets one-by-one to free memory after each fit.

        Examples
        --------
        >>> regressor = TabPFNGRNRegressor(tf_names, target_genes)
        >>> regressor.fit(X, y)
        >>> # After fit, we only need attention weights for edge scores
        >>> regressor.cleanup_model()  # Free GPU memory
        """
        import gc

        if target_name is None:
            # Delete all models
            for model in self.target_models_.values():
                # Explicitly delete model's internal structures
                if hasattr(model, 'models_'):
                    model.models_.clear()
                if hasattr(model, 'device_'):
                    delattr(model, 'device_')
                if hasattr(model, 'devices_'):
                    delattr(model, 'devices_')
            self.target_models_.clear()
        else:
            # Delete specific target model
            if target_name in self.target_models_:
                model = self.target_models_[target_name]
                # Explicitly delete model's internal structures
                if hasattr(model, 'models_'):
                    model.models_.clear()
                if hasattr(model, 'device_'):
                    delattr(model, 'device_')
                if hasattr(model, 'devices_'):
                    delattr(model, 'devices_')
                del self.target_models_[target_name]

        # Force garbage collection and CUDA cache clear
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
