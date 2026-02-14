"""Unified baseline runner for GRN inference methods.

This module provides a consistent interface for running all baseline methods:
- Correlation (Pearson correlation)
- Mutual Information
- GENIE3 (Random Forest)
- GRNBoost2 (Gradient Boosting)

All methods use the same preprocessing pipeline and evaluation framework,
eliminating ~500 lines of duplication from grn_performance_analysis.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tabpfn.grn.baseline_models import (
    SklearnForestWrapper,
    LinearRegressionWrapper,
    TabPFNWrapper,
)
from tabpfn.grn.pipeline import GRNDataPipeline, GRNPreparedData
from tabpfn.grn.utils import evaluate_and_format_results

if TYPE_CHECKING:
    from tabpfn.grn.evaluation import GoldStandardType


class GRNBaselineRunner:
    """Unified runner for GRN baseline methods.

    This class provides a consistent interface for running multiple
    GRN inference methods with the same preprocessing and evaluation.

    Parameters
    ----------
    normalization : str, default='zscore'
        Type of normalization to apply

    Examples
    --------
    >>> runner = GRNBaselineRunner()
    >>> result = runner.run_method(
    ...     method="genie3",
    ...     expression=expression,
    ...     gene_names=gene_names,
    ...     tf_names=tf_names,
    ...     gold_standard=gold_standard,
    ...     dataset_name="DREAM4_10_1"
    ... )
    """

    def __init__(self, normalization: str = "zscore"):
        """Initialize the baseline runner.

        Parameters
        ----------
        normalization : str
            Type of normalization to apply
        """
        self.pipeline = GRNDataPipeline(normalization=normalization)

    def run_method(
        self,
        method: str,
        expression: np.ndarray,
        gene_names: list[str],
        tf_names: list[str],
        gold_standard: "GoldStandardType",
        dataset_name: str,
        target_genes: list[str] | None = None,
        max_targets: int | None = None,
        n_estimators: int = 100,
        random_state: int = 42,
        attention_aggregation: str = "mean",
        edge_score_strategy: str = "self_attention",
    ) -> dict:
        """Run a GRN inference method with unified preprocessing and evaluation.

        This single method replaces run_correlation_baseline(),
        run_genie3_baseline(), run_grnboost2_baseline(),
        run_mutual_information_baseline(), and run_tabpfn().

        Parameters
        ----------
        method : str
            One of: 'correlation', 'mutual_info', 'genie3', 'grnboost2', 'tabpfn'
        expression : np.ndarray
            Gene expression matrix (samples x genes)
        gene_names : list[str]
            All gene names
        tf_names : list[str]
            Transcription factor names
        gold_standard : GoldStandardType
            True regulatory edges
        dataset_name : str
            Name of dataset for reporting
        target_genes : list[str], optional
            Specific target genes to use
        max_targets : int, optional
            Limit to first N targets
        n_estimators : int
            Number of estimators (for forest methods and TabPFN)
        random_state : int
            Random seed
        attention_aggregation : str
            Method to aggregate attention (for TabPFN only, default "mean")
        edge_score_strategy : str
            Method to extract edge scores (for TabPFN only, default "self_attention")

        Returns
        -------
        results : dict
            Dictionary with dataset, method, metrics, edge_scores
        """
        # 1. Prepare data using unified pipeline
        prepared = self.pipeline.prepare_data(
            expression=expression,
            gene_names=gene_names,
            tf_names=tf_names,
            gold_standard=gold_standard,
            target_genes=target_genes,
            max_targets=max_targets,
        )

        # 2. Run the specified method
        if method == "correlation":
            edge_scores, training_time = self._run_correlation(prepared)
        elif method == "mutual_info":
            edge_scores, training_time = self._run_mutual_information(prepared)
        elif method == "genie3":
            edge_scores, training_time = self._run_forest_method(
                prepared=prepared,
                n_estimators=n_estimators,
                random_state=random_state,
                estimator_type="RandomForest",
            )
        elif method == "grnboost2":
            edge_scores, training_time = self._run_forest_method(
                prepared=prepared,
                n_estimators=n_estimators,
                random_state=random_state,
                estimator_type="GradientBoosting",
            )
        elif method == "tabpfn":
            edge_scores, training_time = self._run_pfn_method(
                prepared=prepared,
                n_estimators=n_estimators,
                random_state=random_state,
                attention_aggregation=attention_aggregation,
                edge_score_strategy=edge_score_strategy,
            )
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Must be one of: 'correlation', 'mutual_info', 'genie3', 'grnboost2', 'tabpfn'"
            )

        # 3. Evaluate using unified utility
        return evaluate_and_format_results(
            edge_scores=edge_scores,
            gold_standard=prepared.gold_standard,
            dataset_name=dataset_name,
            method_name=method.upper().replace("_", " "),
            training_time=training_time,
        )

    def _run_correlation(
        self,
        prepared: GRNPreparedData,
    ) -> tuple[dict[tuple[str, str], float], float]:
        """Run correlation baseline.

        Parameters
        ----------
        prepared : GRNPreparedData
            Prepared data from pipeline

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> correlation score
        training_time : float
            Training time (0.0 for correlation)
        """
        from tabpfn.grn.utils import create_edge_score_dict

        # Compute correlation between TFs and targets
        # CRITICAL: When a target gene is also a TF, exclude it from input features
        # to prevent information leakage (self-correlation = 1.0)

        n_tfs = prepared.X.shape[1]
        n_targets = prepared.y.shape[1]

        # Initialize score matrix
        tf_target_correlations = np.zeros((n_tfs, n_targets))

        # For each target gene, compute correlation excluding self if needed
        for target_idx, target_name in enumerate(prepared.target_genes):
            # Get TF indices to use (exclude this target if it's also a TF)
            tf_indices_to_use = []
            for tf_idx, tf_name in enumerate(prepared.tf_names):
                if tf_name != target_name:  # Exclude self
                    tf_indices_to_use.append(tf_idx)

            # Compute correlation between selected TFs and this target
            if tf_indices_to_use:
                X_for_target = prepared.X[:, tf_indices_to_use]
                y_target = prepared.y[:, target_idx]
                corrs = [np.corrcoef(X_for_target[:, i], y_target)[0, 1]
                         for i in range(X_for_target.shape[1])]
                # Store in the correct positions
                for i, tf_idx in enumerate(tf_indices_to_use):
                    tf_target_correlations[tf_idx, target_idx] = abs(corrs[i])

        # Use absolute correlation as edge scores
        edge_scores = create_edge_score_dict(
            tf_names=prepared.tf_names,
            target_genes=prepared.target_genes,
            scores=tf_target_correlations,
            skip_self_edges=True,  # Skip self-edges (even if zero)
        )

        return edge_scores, 0.0  # Correlation is instant

    def _run_mutual_information(
        self,
        prepared: GRNPreparedData,
    ) -> tuple[dict[tuple[str, str], float], float]:
        """Run mutual information baseline.

        Parameters
        ----------
        prepared : GRNPreparedData
            Prepared data from pipeline

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> MI score
        training_time : float
            Training time (near 0.0 for MI)
        """
        import time
        from sklearn.feature_selection import mutual_info_regression

        start_time = time.time()

        edge_scores = {}
        for j, tgt in enumerate(prepared.target_genes):
            # Exclude target gene from input features to prevent leakage
            tf_indices_to_use = []
            tf_names_to_use = []
            for tf_idx, tf_name in enumerate(prepared.tf_names):
                if tf_name != tgt:
                    tf_indices_to_use.append(tf_idx)
                    tf_names_to_use.append(tf_name)

            X_for_target = prepared.X[:, tf_indices_to_use] if tf_indices_to_use else prepared.X
            mi_scores = mutual_info_regression(
                X_for_target,
                prepared.y[:, j],
                random_state=42,
            )
            for i, tf_name in enumerate(tf_names_to_use if tf_indices_to_use else prepared.tf_names):
                edge_scores[(tf_name, tgt)] = mi_scores[i]

        # Normalize to [0, 1]
        max_score = max(edge_scores.values()) if edge_scores else 1.0
        edge_scores = {k: v / max_score for k, v in edge_scores.items()}

        training_time = time.time() - start_time
        return edge_scores, training_time

    def _run_pfn_method(
        self,
        prepared: GRNPreparedData,
        n_estimators: int,
        random_state: int,
        attention_aggregation: str,
        edge_score_strategy: str,
    ) -> tuple[dict[tuple[str, str], float], float]:
        """Run TabPFN GRN inference method.

        Uses TabPFNWrapper which properly handles target exclusion:
        When a target gene is also a TF, it is excluded from input features
        to prevent information leakage (self-correlation = 1.0).

        Parameters
        ----------
        prepared : GRNPreparedData
            Prepared data from pipeline
        n_estimators : int
            Number of TabPFN estimators per target
        random_state : int
            Random seed
        attention_aggregation : str
            Method to aggregate attention (passed but not used by TabPFNWrapper)
        edge_score_strategy : str
            Method to extract edge scores (passed but not used by TabPFNWrapper)

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> importance score
        training_time : float
            Time taken to fit models
        """
        import time

        # Import TabPFNWrapper which has proper target exclusion via prepare_target_features()
        from tabpfn.grn.baseline_models import TabPFNWrapper

        # Extract tf_names and target_genes from prepared data
        tf_names = prepared.tf_names
        target_genes = prepared.target_genes

        # Create TabPFNWrapper with specific parameters
        # Note: TabPFNWrapper uses prepare_target_features() which properly excludes
        # each target gene from its own input features when it's also a TF
        wrapper = TabPFNWrapper(
            n_estimators=n_estimators,
            attention_aggregation=attention_aggregation,
            edge_score_strategy=edge_score_strategy,
            device="auto",
            random_state=random_state,
        )

        # Fit model with tf_names and target_genes
        # TabPFNWrapper.fit() uses prepare_target_features() for proper target exclusion
        start_time = time.time()
        wrapper.fit(
            prepared.X,
            prepared.y,
            tf_names=tf_names,
            target_genes=target_genes,
        )
        training_time = time.time() - start_time

        # Extract edge scores
        edge_scores = wrapper.get_edge_scores(
            tf_names=tf_names,
            target_genes=target_genes,
        )

        return edge_scores, training_time

    def run_tabpfn_multiple_strategies(
        self,
        expression: np.ndarray,
        gene_names: list[str],
        tf_names: list[str],
        gold_standard: "GoldStandardType",
        dataset_name: str,
        target_genes: list[str] | None = None,
        max_targets: int | None = None,
        n_estimators: int = 1,
        random_state: int = 42,
        attention_aggregation: str = "mean",
        edge_score_strategies: list[str] | None = None,
    ) -> dict[str, dict]:
        """Run TabPFN with multiple edge score strategies using per-target processing.

        Uses fit_one_target_gene() and get_edge_score_one_target_gene() to handle
        memory-efficient per-target processing with proper edge score extraction
        before model cleanup.

        This approach fixes gradient_rollout KeyError errors by extracting edge scores
        for each target immediately after fitting (before model cleanup).

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
        dataset_name : str
            Name of dataset for reporting
        target_genes : list[str], optional
            Specific target genes to use
        max_targets : int, optional
            Limit to first N targets
        n_estimators : int
            Number of TabPFN estimators per target
        random_state : int
            Random seed
        attention_aggregation : str
            Method to aggregate attention
        edge_score_strategies : list[str], optional
            List of edge score strategies to evaluate.
            If None, uses all 7 strategies: ['self_attention', 'tf_to_target',
            'target_to_tf', 'combined', 'combined_best', 'sequential_rollout',
            'gradient_rollout']

        Returns
        -------
        results : dict
            Dictionary mapping strategy name to result dict with metrics
        """
        import time
        import gc

        if edge_score_strategies is None:
            edge_score_strategies = [
                'self_attention', 'tf_to_target', 'target_to_tf',
                'combined', 'combined_best', 'sequential_rollout', 'gradient_rollout'
            ]

        # 1. Prepare data using unified pipeline
        prepared = self.pipeline.prepare_data(
            expression=expression,
            gene_names=gene_names,
            tf_names=tf_names,
            gold_standard=gold_standard,
            target_genes=target_genes,
            max_targets=max_targets,
        )

        # 2. Create TabPFNWrapper
        from tabpfn.grn.baseline_models import TabPFNWrapper

        wrapper = TabPFNWrapper(
            n_estimators=n_estimators,
            attention_aggregation=attention_aggregation,
            edge_score_strategy=edge_score_strategies[0],  # Default strategy
            device="auto",
            random_state=random_state,
            keep_model=True,  # Keep model for edge score extraction, cleanup manually
        )

        # Initialize edge score dictionaries for each strategy
        strategy_edge_scores = {strategy: {} for strategy in edge_score_strategies}

        start_time = time.time()

        # Precompute per-target features with target exclusion
        from tabpfn.grn.utils import compute_target_feature_indices
        per_target_features = {}
        for target_idx, target_name in enumerate(prepared.target_genes):
            tf_indices, tf_names_for_target, _ = compute_target_feature_indices(
                tf_names=prepared.tf_names,
                target_name=target_name,
            )
            per_target_features[target_idx] = (
                prepared.X[:, tf_indices],
                tf_names_for_target,
            )

        # 3. Outer loop: For each target gene
        for target_idx, target_name in enumerate(prepared.target_genes):
            y_target = prepared.y[:, target_idx]
            X_for_target, tf_names_for_target = per_target_features[target_idx]

            # Fit this target's model
            wrapper.fit_one_target_gene(
                target_idx=target_idx,
                target_name=target_name,
                X_for_target=X_for_target,
                y_target=y_target,
                tf_names_for_target=tf_names_for_target,
            )

            # Inner loop: Extract edge scores for each strategy BEFORE cleanup
            for strategy in edge_score_strategies:
                target_edge_scores = wrapper.get_edge_score_one_target_gene(
                    target_name=target_name,
                    edge_score_strategy=strategy,
                )
                # Add to the strategy's edge score dictionary
                strategy_edge_scores[strategy].update(target_edge_scores)

            # Manually cleanup after extracting all edge scores for this target
            wrapper._regressors[target_name].cleanup_model()
            # CRITICAL: Delete the regressor from _regressors dict to free memory
            # The regressor contains attention_weights_, X_, y_ and other tensors
            # that accumulate in GPU memory even after model cleanup
            del wrapper._regressors[target_name]
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

        training_time = time.time() - start_time

        # 4. Evaluate each strategy's edge scores
        results = {}
        for strategy in edge_score_strategies:
            result = evaluate_and_format_results(
                edge_scores=strategy_edge_scores[strategy],
                gold_standard=prepared.gold_standard,
                dataset_name=dataset_name,
                method_name=f"TABPFN ({strategy})",
                training_time=training_time,
            )
            results[strategy] = result

        return results

    def _run_forest_method(
        self,
        prepared: GRNPreparedData,
        n_estimators: int = 100,
        random_state: int = 42,
        estimator_type: str = "RandomForest",
    ) -> tuple[dict[tuple[str, str], float], float]:
        """Run forest-based method (GENIE3 or GRNBoost2).

        Parameters
        ----------
        prepared : GRNPreparedData
            Prepared data from pipeline
        n_estimators : int
            Number of estimators
        random_state : int
            Random seed
        estimator_type : str
            Type of forest ('RandomForest' or 'GradientBoosting')

        Returns
        -------
        edge_scores : dict
            Dictionary mapping (tf, target) -> importance score
        training_time : float
            Time taken to fit models
        """
        import time
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        # Select estimator class and kwargs
        if estimator_type == "GradientBoosting":
            EstimatorClass = GradientBoostingRegressor
            estimator_kwargs = {
                "learning_rate": 0.1,
                "max_depth": 3,
            }
            method_name = "GRNBoost2"
        else:
            EstimatorClass = RandomForestRegressor
            estimator_kwargs = {"max_features": "sqrt"}
            method_name = "GENIE3"

        # Create wrapper using baseline_models.py
        wrapper = SklearnForestWrapper(
            estimator_class=EstimatorClass,
            estimator_kwargs=estimator_kwargs,
            n_estimators=n_estimators,
            random_state=random_state,
        )

        # Fit model
        start_time = time.time()
        wrapper.fit(
            prepared.X,
            prepared.y,
            tf_names=prepared.tf_names,
            target_genes=prepared.target_genes,
        )
        training_time = time.time() - start_time

        # Extract edge scores using method we added to baseline_models.py
        edge_scores = wrapper.get_edge_scores(
            tf_names=prepared.tf_names,
            target_genes=prepared.target_genes,
        )

        return edge_scores, training_time
