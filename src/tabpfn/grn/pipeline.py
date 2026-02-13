"""Unified preprocessing and evaluation pipeline for GRN inference.

This module provides a unified interface for:
1. Loading and preprocessing datasets
2. Creating train/test splits
3. Running GRN inference methods
4. Evaluating results with consistent metrics

Key Design Principles:
- Decouple data preparation from model training
- Reuse existing baseline_models.py classes
- Use utils.evaluate_and_format_results() for consistency
- Support multiple datasets (DREAM4, DREAM5) seamlessly
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from tabpfn.grn.preprocessing import GRNPreprocessor
from tabpfn.grn.utils import compute_target_feature_indices

if TYPE_CHECKING:
    from tabpfn.grn.evaluation import GoldStandardType


@dataclass
class GRNPreparedData:
    """Container for prepared GRN data.

    This provides a unified interface that all baseline methods
    can use, eliminating the need for duplicate data preparation logic.

    Attributes
    ----------
    X_train : np.ndarray
        TF expression matrix for training (n_samples, n_TFs)
    X_test : np.ndarray or None
        TF expression matrix for testing (n_test_samples, n_TFs)
    y_train : np.ndarray
        Target expression matrix for training (n_samples, n_targets)
    y_test : np.ndarray or None
        Target expression matrix for testing (n_test_samples, n_targets)
    X : np.ndarray
        Full TF expression matrix (n_samples, n_TFs)
    y : np.ndarray
        Full target expression matrix (n_samples, n_targets)
    tf_names : list[str]
        Transcription factor names
    target_genes : list[str]
        Target gene names
    gold_standard : set[tuple[str, str]]
        Filtered gold standard network
    preprocessor : GRNPreprocessor
        Fitted preprocessor (for transforming new data)
    """
    X_train: np.ndarray
    X_test: np.ndarray | None
    y_train: np.ndarray
    y_test: np.ndarray | None
    X: np.ndarray
    y: np.ndarray
    tf_names: list[str]
    target_genes: list[str]
    gold_standard: set[tuple[str, str]] | Any
    preprocessor: GRNPreprocessor


class GRNDataPipeline:
    """Unified pipeline for GRN inference preprocessing and evaluation.

    This class handles:
    - Data loading (DREAM4, DREAM5, custom)
    - Preprocessing (normalization, train/test split)
    - Target subset selection (for large datasets)
    - Gold standard filtering

    The pipeline returns prepared data that can be used with ANY GRN inference
    method (TabPFN, GENIE3, GRNBoost2, Correlation, MI).

    Parameters
    ----------
    normalization : str, default='zscore'
        Type of normalization ('zscore', 'log', 'quantile', 'none')
    test_size : float, default=0.0
        Fraction of data to use for testing (0.0 = use all data)
    random_state : int, default=42
        Random seed for reproducibility

    Examples
    --------
    >>> pipeline = GRNDataPipeline(normalization='zscore')
    >>> prepared = pipeline.prepare_data(
    ...     expression=expression,
    ...     gene_names=gene_names,
    ...     tf_names=tf_names,
    ...     gold_standard=gold_standard,
    ... )
    >>> print(f"Prepared {len(prepared.target_genes)} targets")
    """

    def __init__(
        self,
        normalization: str = "zscore",
        test_size: float = 0.0,
        random_state: int = 42,
    ):
        """Initialize the GRN data pipeline.

        Parameters
        ----------
        normalization : str
            Type of normalization to apply
        test_size : float
            Fraction of data to use for testing
        random_state : int
            Random seed for reproducibility
        """
        self.normalization = normalization
        self.test_size = test_size
        self.random_state = random_state

    def prepare_data(
        self,
        expression: np.ndarray,
        gene_names: list[str],
        tf_names: list[str],
        gold_standard: "GoldStandardType",
        target_genes: list[str] | None = None,
        max_targets: int | None = None,
    ) -> GRNPreparedData:
        """Prepare data for GRN inference.

        This single method replaces duplicated data preparation logic
        across all baseline functions in grn_performance_analysis.py.

        Parameters
        ----------
        expression : np.ndarray
            Gene expression matrix (samples x genes)
        gene_names : list[str]
            All gene names
        tf_names : list[str]
            Transcription factor names
        gold_standard : GoldStandardType
            True regulatory edges
        target_genes : list[str], optional
            Specific target genes to use (if None, uses all non-TF genes)
        max_targets : int, optional
            Limit to first N targets (for efficiency on large datasets)

        Returns
        -------
        GRNPreparedData
            Named tuple with X_train, X_test, y_train, y_test, X, y,
            tf_names, target_genes, gold_standard, preprocessor
        """
        # 1. Create preprocessor and fit_transform
        preprocessor = GRNPreprocessor(normalization=self.normalization)
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # 2. Get target gene names from preprocessor
        all_target_genes = preprocessor.get_target_names()

        # 3. Filter to requested targets
        if target_genes is None:
            target_genes = all_target_genes
        if max_targets is not None and len(target_genes) > max_targets:
            target_genes = target_genes[:max_targets]

        # 4. Filter y to selected targets
        target_indices_selected = [
            i for i, tgt in enumerate(all_target_genes)
            if tgt in target_genes
        ]
        y_filtered = y[:, target_indices_selected]

        # 5. Filter gold standard
        filtered_gold_standard = self._filter_gold_standard(
            gold_standard, tf_names, target_genes
        )

        # 6. Apply train/test split if requested
        if self.test_size > 0:
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_filtered,
                test_size=self.test_size,
                random_state=self.random_state
            )
        else:
            X_train, X_test = X, None
            y_train, y_test = y_filtered, None

        return GRNPreparedData(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X=X,
            y=y_filtered,
            tf_names=tf_names,
            target_genes=target_genes,
            gold_standard=filtered_gold_standard,
            preprocessor=preprocessor,
        )

    def _filter_gold_standard(
        self,
        gold_standard: "GoldStandardType",
        tf_names: list[str],
        target_genes: list[str],
    ) -> set[tuple[str, str]]:
        """Filter gold standard to only include relevant edges.

        Parameters
        ----------
        gold_standard : GoldStandardType
            True regulatory edges (set of tuples or DataFrame)
        tf_names : list[str]
            Transcription factor names
        target_genes : list[str]
            Target gene names

        Returns
        -------
        filtered_gold_standard : set[tuple[str, str]]
            Gold standard filtered to only include edges where
            both TF and target are in our selected genes
        """
        tf_set = set(tf_names)
        target_set = set(target_genes)

        if isinstance(gold_standard, set):
            return {
                (tf, tgt) for tf, tgt in gold_standard
                if tf in tf_set and tgt in target_set
            }
        else:  # DataFrame
            import pandas as pd
            gs_set = set()
            for _, row in gold_standard.iterrows():
                # Handle different column formats
                if "tf" in row and "target" in row:
                    tf, tgt = row["tf"], row["target"]
                    weight = row.get("weight", 1)  # Default to 1 if no weight column
                elif "TF" in row and "Target" in row:
                    tf, tgt = row["TF"], row["Target"]
                    weight = row.get("weight", 1)
                elif "source" in row and "target" in row:
                    tf, tgt = row["source"], row["target"]
                    weight = row.get("weight", 1)
                else:
                    tf, tgt = row.iloc[0], row.iloc[1]
                    weight = row.get("weight", 1)

                # Filter: must be in gene sets AND have weight > 0
                if tf in tf_set and tgt in target_set and weight > 0:
                    gs_set.add((tf, tgt))
            return gs_set
