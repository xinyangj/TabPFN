"""Preprocessing utilities for Gene Regulatory Network (GRN) inference.

This module provides preprocessing functionality specific to GRN analysis,
including transcription factor (TF) identification, expression normalization,
and feature engineering for regulatory relationship prediction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


class GRNPreprocessor:
    """Preprocess gene expression data for GRN inference.

    This preprocessor handles:
    - Identification of TFs vs target genes
    - Expression normalization (z-score, log, quantile)
    - Feature engineering (interaction features, prior knowledge)

    Parameters
    ----------
    normalization : str, default='zscore'
        Type of normalization to apply. Options:
        - 'zscore': Standard score normalization (mean=0, std=1)
        - 'log': Log transformation (log1p)
        - 'quantile': Quantile transformation
        - 'none': No normalization

    add_interaction_features : bool, default=False
        Whether to add pairwise TF-TF interaction features.

    Examples
    --------
    >>> preprocessor = GRNPreprocessor(normalization='zscore')
    >>> X, y, tf_indices, target_indices = preprocessor.fit_transform(
    ...     expression, gene_names, tf_names
    ... )
    >>> print(f"TF features shape: {X.shape}")
    >>> print(f"Target genes shape: {y.shape}")
    """

    def __init__(
        self,
        normalization: str = "zscore",
        add_interaction_features: bool = False,
    ) -> None:
        """Initialize the GRN preprocessor.

        Parameters
        ----------
        normalization : str, default='zscore'
            Type of normalization to apply.

        add_interaction_features : bool, default=False
            Whether to add pairwise TF-TF interaction features.
        """
        if normalization not in ["zscore", "log", "quantile", "none"]:
            raise ValueError(
                f"normalization must be 'zscore', 'log', 'quantile', or 'none', "
                f"got '{normalization}'"
            )

        self.normalization = normalization
        self.add_interaction_features = add_interaction_features

        # Fitted parameters
        self.mean_tf = None
        self.std_tf = None
        self.mean_target = None
        self.std_target = None
        self.tf_names_ = None
        self.target_names_ = None

    def fit_transform(
        self,
        expression: np.ndarray,
        gene_names: list[str],
        tf_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
        """Preprocess expression data for GRN inference.

        Identifies TFs and targets, normalizes expression, and creates
        feature/target matrices.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix of shape (n_samples, n_genes).

        gene_names : list[str]
            Gene names corresponding to columns of expression matrix.

        tf_names : list[str]
            List of known transcription factor names (subset of gene_names).

        Returns
        -------
        X : np.ndarray
            Feature matrix for input (TF expression).
            Shape: (n_samples, n_TFs) or (n_samples, n_TFs + n_interactions)
            if add_interaction_features=True.

        y : np.ndarray
            Target gene expression matrix.
            Shape: (n_samples, n_targets)

        tf_indices : list[int]
            Indices of TFs in the original gene_names list.

        target_indices : list[int]
            Indices of target genes in the original gene_names list.

        Examples
        --------
        >>> import numpy as np
        >>> expression = np.random.randn(100, 100)  # 100 samples, 100 genes
        >>> gene_names = [f"GENE_{i}" for i in range(100)]
        >>> tf_names = gene_names[:10]  # First 10 are TFs
        >>> preprocessor = GRNPreprocessor()
        >>> X, y, tf_idx, target_idx = preprocessor.fit_transform(
        ...     expression, gene_names, tf_names
        ... )
        """
        # Validate inputs
        if expression.ndim != 2:
            raise ValueError(
                f"expression must be 2D, got shape {expression.shape}"
            )
        if len(gene_names) != expression.shape[1]:
            raise ValueError(
                f"Number of gene names ({len(gene_names)}) must match "
                f"number of columns in expression ({expression.shape[1]})"
            )

        # Identify TFs and targets
        tf_set = set(tf_names)
        tf_indices = []
        target_indices = []

        for i, gene in enumerate(gene_names):
            if gene in tf_set:
                tf_indices.append(i)
            else:
                target_indices.append(i)

        if not tf_indices:
            raise ValueError("No TFs found in gene_names")
        if not target_indices:
            raise ValueError("No target genes found in gene_names")

        # Store for later use
        self.tf_names_ = [gene_names[i] for i in tf_indices]
        self.target_names_ = [gene_names[i] for i in target_indices]

        # Extract TF and target expression
        tf_expression = expression[:, tf_indices]
        target_expression = expression[:, target_indices]

        # Normalize expression (normalize TF and target independently)
        tf_expression_norm = self._normalize(tf_expression, fit=True, suffix="tf")
        target_expression_norm = self._normalize(target_expression, fit=True, suffix="target")

        # Create feature matrix
        X = tf_expression_norm
        if self.add_interaction_features:
            X = self._add_interaction_features(X)

        y = target_expression_norm

        return X, y, tf_indices, target_indices

    def _normalize(
        self, data: np.ndarray, fit: bool, suffix: str = ""
    ) -> np.ndarray:
        """Normalize expression data.

        Parameters
        ----------
        data : np.ndarray
            Expression data to normalize.

        fit : bool
            If True, fit normalization parameters on this data.
            If False, use previously fitted parameters.

        suffix : str, default=""
            Suffix to distinguish TF and target normalization parameters.

        Returns
        -------
        normalized : np.ndarray
            Normalized expression data.
        """
        if self.normalization == "none":
            return data

        if self.normalization == "zscore":
            mean_attr = f"mean_{suffix}" if suffix else "mean_"
            std_attr = f"std_{suffix}" if suffix else "std_"

            if fit:
                setattr(self, mean_attr, np.mean(data, axis=0))
                std = np.std(data, axis=0)
                # Avoid division by zero
                std[std == 0] = 1.0
                setattr(self, std_attr, std)

            mean = getattr(self, mean_attr)
            std = getattr(self, std_attr)
            normalized = (data - mean) / std
            return normalized

        elif self.normalization == "log":
            # Log1p transformation (more stable than log for small values)
            return np.log1p(data - data.min())  # Shift to be non-negative

        elif self.normalization == "quantile":
            # Simple quantile normalization
            from scipy.stats import rankdata

            normalized = np.zeros_like(data)
            for col in range(data.shape[1]):
                ranks = rankdata(data[:, col])
                normalized[:, col] = ranks / (len(ranks) + 1)
            return normalized

        else:
            return data

    def _add_interaction_features(self, X: np.ndarray) -> np.ndarray:
        """Add pairwise TF-TF interaction features.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix of shape (n_samples, n_TFs).

        Returns
        -------
        X_with_interactions : np.ndarray
            Feature matrix with interaction features added.
            Shape: (n_samples, n_TFs + n_TFs * (n_TFs - 1) / 2)
        """
        n_samples, n_tfs = X.shape
        interaction_features = []

        for i in range(n_tfs):
            for j in range(i + 1, n_tfs):
                interaction = X[:, i] * X[:, j]
                interaction_features.append(interaction)

        X_with_interactions = np.hstack([X] + [f[:, None] for f in interaction_features])
        return X_with_interactions

    def get_tf_names(self) -> list[str] | None:
        """Get TF names after fitting.

        Returns
        -------
        tf_names : list[str] or None
            List of TF names, or None if not fitted.
        """
        return self.tf_names_

    def get_target_names(self) -> list[str] | None:
        """Get target gene names after fitting.

        Returns
        -------
        target_names : list[str] or None
            List of target gene names, or None if not fitted.
        """
        return self.target_names_

    def transform(
        self, expression: np.ndarray, gene_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform new expression data using fitted parameters.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix of shape (n_samples, n_genes).

        gene_names : list[str]
            Gene names corresponding to columns of expression matrix.

        Returns
        -------
        X : np.ndarray
            Normalized TF expression features.

        y : np.ndarray
            Normalized target gene expression.

        Raises
        ------
        ValueError
            If the preprocessor hasn't been fitted yet.
        """
        if self.tf_names_ is None or self.target_names_ is None:
            raise ValueError("Preprocessor must be fitted before transform")

        # Find TF and target indices
        tf_to_idx = {name: i for i, name in enumerate(gene_names)}
        target_to_idx = {name: i for i, name in enumerate(gene_names)}

        tf_indices = [tf_to_idx[name] for name in self.tf_names_]
        target_indices = [target_to_idx[name] for name in self.target_names_]

        # Extract and normalize
        tf_expression = expression[:, tf_indices]
        target_expression = expression[:, target_indices]

        tf_expression_norm = self._normalize(tf_expression, fit=False, suffix="tf")
        target_expression_norm = self._normalize(target_expression, fit=False, suffix="target")

        X = tf_expression_norm
        if self.add_interaction_features:
            X = self._add_interaction_features(X)

        y = target_expression_norm

        return X, y
