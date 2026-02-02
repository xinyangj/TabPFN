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
from sklearn.base import BaseEstimator

if TYPE_CHECKING:
    import networkx as nx

from tabpfn import TabPFNRegressor
from tabpfn.grn.attention_extractor import AttentionExtractor, EdgeScoreComputer


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
        """
        self.tf_names = tf_names
        self.target_genes = target_genes
        self.n_estimators = n_estimators
        self.attention_aggregation = attention_aggregation
        self.edge_score_strategy = edge_score_strategy
        self.device = device
        self.n_jobs = n_jobs

        # Fitted attributes
        self.target_models_: dict[str, TabPFNRegressor] = {}
        self.attention_weights_: dict[str, dict] = {}
        self.edge_scores_: dict[tuple[str, str], float] = {}

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

        # Train one model per target
        self.target_models_ = {}
        self.attention_weights_ = {}

        for target_idx, target_name in enumerate(self.target_genes):
            # Train model for this target
            model = TabPFNRegressor(
                n_estimators=self.n_estimators,
                device=self.device,
            )
            model.fit(X, y[:, target_idx])
            self.target_models_[target_name] = model

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
            # Compute aggregated attention for this target
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
