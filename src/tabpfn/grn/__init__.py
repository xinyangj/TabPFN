"""TabPFN Gene Regulatory Network (GRN) Analysis Module.

This module extends TabPFN for gene regulatory network inference by:
1. Using TabPFN to predict gene expression from transcription factors (TFs)
2. Extracting attention weights to discover GRN edges (TF-target relationships)
3. Visualization tools for analyzing GRNs

Note: This approach uses TabPFN as a frozen foundation model with in-context learning.
No fine-tuning or weight updates are performed.

Usage:
    # Run multiple TabPFN strategies with a single fit
    runner = GRNBaselineRunner()
    results = runner.run_tabpfn_multiple_strategies(
        expression=expression,
        gene_names=gene_names,
        tf_names=tf_names,
        gold_standard=gold_standard,
        dataset_name="DREAM4_10_1",
        edge_score_strategies=['self_attention', 'tf_to_target', 'target_to_tf'],
    )
"""

from tabpfn.grn.attention_extractor import (
    AttentionExtractor,
    EdgeScoreComputer,
    GradientAttentionExtractor,
)
from tabpfn.grn.baseline_runner import GRNBaselineRunner
from tabpfn.grn.pipeline import GRNDataPipeline, GRNPreparedData
from tabpfn.grn.preprocessing import GRNPreprocessor
from tabpfn.grn.datasets import DREAMChallengeLoader
from tabpfn.grn.evaluation import (
    compute_aupr,
    compute_auroc,
    compute_f1_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    compute_expression_metrics,
    evaluate_expression_prediction,
    evaluate_grn,
)
from tabpfn.grn.baseline_models import TabPFNWrapper
from tabpfn.grn.grn_regressor import TabPFNGRNRegressor
from tabpfn.grn.visualization import (
    AttentionHeatmapVisualizer,
    EdgeScoreVisualizer,
    GRNNetworkVisualizer,
    create_evaluation_summary_plot,
)

__all__ = [
    "AttentionExtractor",
    "EdgeScoreComputer",
    "GradientAttentionExtractor",
    "DREAMChallengeLoader",
    "GRNPreprocessor",
    "TabPFNWrapper",
    "TabPFNGRNRegressor",
    "GRNDataPipeline",
    "GRNPreparedData",
    "GRNBaselineRunner",
    "compute_aupr",
    "compute_auroc",
    "compute_f1_at_k",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_expression_metrics",
    "evaluate_expression_prediction",
    "evaluate_grn",
    "AttentionHeatmapVisualizer",
    "EdgeScoreVisualizer",
    "GRNNetworkVisualizer",
    "create_evaluation_summary_plot",
    "prepare_target_features",
    "SklearnForestWrapper",
    "LinearRegressionWrapper",
]
