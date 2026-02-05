"""TabPFN Gene Regulatory Network (GRN) Analysis Module.

This module extends TabPFN for gene regulatory network inference by:
1. Using TabPFN to predict gene expression from transcription factors (TFs)
2. Extracting attention weights to discover GRN edges (TF-target relationships)

Note: This approach uses TabPFN as a frozen foundation model with in-context learning.
No fine-tuning or weight updates are performed.
"""

from tabpfn.grn.attention_extractor import AttentionExtractor, EdgeScoreComputer
from tabpfn.grn.datasets import DREAMChallengeLoader
from tabpfn.grn.evaluation import (
    compute_aupr,
    compute_auroc,
    compute_expression_metrics,
    compute_f1_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
    evaluate_expression_prediction,
    evaluate_grn,
)
from tabpfn.grn.grn_regressor import TabPFNGRNRegressor
from tabpfn.grn.preprocessing import GRNPreprocessor
from tabpfn.grn.visualization import (
    AttentionHeatmapVisualizer,
    EdgeScoreVisualizer,
    GRNNetworkVisualizer,
    create_evaluation_summary_plot,
)

__all__ = [
    "AttentionExtractor",
    "EdgeScoreComputer",
    "DREAMChallengeLoader",
    "GRNPreprocessor",
    "TabPFNGRNRegressor",
    "compute_auroc",
    "compute_aupr",
    "compute_expression_metrics",
    "compute_precision_at_k",
    "compute_recall_at_k",
    "compute_f1_at_k",
    "evaluate_expression_prediction",
    "evaluate_grn",
    "GRNNetworkVisualizer",
    "AttentionHeatmapVisualizer",
    "EdgeScoreVisualizer",
    "create_evaluation_summary_plot",
]
