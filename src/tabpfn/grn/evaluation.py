"""Evaluation metrics for Gene Regulatory Network (GRN) inference.

This module provides standard metrics used in the DREAM challenges for
evaluating GRN inference methods, including AUROC, AUPR, and Precision@k.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

if TYPE_CHECKING:
    import networkx as nx
    import pandas as pd


def compute_auroc(
    inferred_edges: dict[tuple[str, str], float] | "nx.DiGraph",
    gold_standard: "pd.DataFrame" | set[tuple[str, str]],
) -> float:
    """Compute Area Under ROC Curve (AUROC) for inferred GRN.

    AUROC measures the ability of the method to distinguish between
    true and false regulatory relationships.

    Parameters
    ----------
    inferred_edges : dict or nx.DiGraph
        Inferred regulatory edges with scores.
        If dict: keys are (tf, target) tuples, values are scores
        If nx.DiGraph: edges must have 'weight' attribute

    gold_standard : pd.DataFrame or set
        Gold standard regulatory network.
        If pd.DataFrame: must have 'tf' and 'target' columns
        If set: set of (tf, target) tuples

    Returns
    -------
    auroc : float
        Area Under ROC Curve, ranges from 0 to 1.
        Higher is better.

    Examples
    --------
    >>> inferred = {('TF1', 'GENE1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> gold_standard = {('TF1', 'GENE1')}
    >>> auroc = compute_auroc(inferred, gold_standard)
    """
    # Convert inputs to consistent format
    if isinstance(inferred_edges, dict):
        # Dict items are ((tf, target), score) tuples
        edge_list = [(tf, target, score) for (tf, target), score in inferred_edges.items()]
    else:  # nx.DiGraph
        edge_list = [
            (u, v, data.get("weight", 1.0))
            for u, v, data in inferred_edges.edges(data=True)
        ]

    if not edge_list:
        return 0.0

    # Create set of true edges
    if isinstance(gold_standard, set):
        true_edges = gold_standard
    else:  # pd.DataFrame
        true_edges = set(zip(gold_standard["tf"], gold_standard["target"]))

    # Get all possible TF-target pairs
    all_pairs = [(tf, tgt) for tf, tgt, _ in edge_list]
    all_tfs = list({tf for tf, _ in all_pairs})
    all_targets = list({tgt for _, tgt in all_pairs})

    # Create binary labels and scores
    y_true = []
    y_score = []

    for tf in all_tfs:
        for target in all_targets:
            is_true = (tf, target) in true_edges
            score = 0.0

            # Find score for this edge
            for tf_inf, tgt_inf, edge_score in edge_list:
                if tf_inf == tf and tgt_inf == target:
                    score = edge_score
                    break

            y_true.append(1 if is_true else 0)
            y_score.append(score)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Compute AUROC
    if len(np.unique(y_true)) < 2:
        # If all labels are the same, return 0.5 (random performance)
        return 0.5

    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc = auc(fpr, tpr)

    return float(auroc)


def compute_aupr(
    inferred_edges: dict[tuple[str, str], float] | "nx.DiGraph",
    gold_standard: "pd.DataFrame" | set[tuple[str, str]],
) -> float:
    """Compute Area Under Precision-Recall Curve (AUPR) for inferred GRN.

    AUPR is particularly important for GRN inference because:
    - GRNs are very sparse (few true edges among many possibilities)
    - AUPR is more informative than AUROC for imbalanced data

    Parameters
    ----------
    inferred_edges : dict or nx.DiGraph
        Inferred regulatory edges with scores

    gold_standard : pd.DataFrame or set
        Gold standard regulatory network

    Returns
    -------
    aupr : float
        Area Under Precision-Recall Curve, ranges from 0 to 1.
        Higher is better.

    Examples
    --------
    >>> inferred = {('TF1', 'GENE1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> gold_standard = {('TF1', 'GENE1')}
    >>> aupr = compute_aupr(inferred, gold_standard)
    """
    # Convert inputs to consistent format
    if isinstance(inferred_edges, dict):
        # Dict items are ((tf, target), score) tuples
        edge_list = [(tf, target, score) for (tf, target), score in inferred_edges.items()]
    else:  # nx.DiGraph
        edge_list = [
            (u, v, data.get("weight", 1.0))
            for u, v, data in inferred_edges.edges(data=True)
        ]

    if not edge_list:
        return 0.0

    # Create set of true edges
    if isinstance(gold_standard, set):
        true_edges = gold_standard
    else:  # pd.DataFrame
        true_edges = set(zip(gold_standard["tf"], gold_standard["target"]))

    # Get all possible TF-target pairs
    all_pairs = [(tf, tgt) for tf, tgt, _ in edge_list]
    all_tfs = list({tf for tf, _ in all_pairs})
    all_targets = list({tgt for _, tgt in all_pairs})

    # Create binary labels and scores
    y_true = []
    y_score = []

    for tf in all_tfs:
        for target in all_targets:
            is_true = (tf, target) in true_edges
            score = 0.0

            # Find score for this edge
            for tf_inf, tgt_inf, edge_score in edge_list:
                if tf_inf == tf and tgt_inf == target:
                    score = edge_score
                    break

            y_true.append(1 if is_true else 0)
            y_score.append(score)

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Compute AUPR
    if len(np.unique(y_true)) < 2:
        # If all labels are the same, return 0.0
        return 0.0

    aupr = average_precision_score(y_true, y_score)

    return float(aupr)


def compute_precision_at_k(
    inferred_edges: dict[tuple[str, str], float] | "nx.DiGraph",
    gold_standard: "pd.DataFrame" | set[tuple[str, str]],
    k: int = 100,
) -> float:
    """Compute precision of top-k predicted edges.

    Precision@k measures the fraction of true edges among the top-k
    highest-scoring predictions. This is particularly relevant for
    experimental validation where researchers can only test a
    limited number of predictions.

    Parameters
    ----------
    inferred_edges : dict or nx.DiGraph
        Inferred regulatory edges with scores

    gold_standard : pd.DataFrame or set
        Gold standard regulatory network

    k : int, default=100
        Number of top predictions to consider

    Returns
    -------
    precision_at_k : float
        Precision@k = (true positives in top k) / k
        Ranges from 0 to 1, higher is better.

    Examples
    --------
    >>> inferred = {('TF1', 'GENE1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> gold_standard = {('TF1', 'GENE1')}
    >>> precision = compute_precision_at_k(inferred, gold_standard, k=10)
    """
    # Convert inputs to consistent format
    if isinstance(inferred_edges, dict):
        # Dict items are ((tf, target), score) tuples
        edge_list = [(tf, target, score) for (tf, target), score in inferred_edges.items()]
    else:  # nx.DiGraph
        edge_list = [
            (u, v, data.get("weight", 1.0))
            for u, v, data in inferred_edges.edges(data=True)
        ]

    if not edge_list:
        return 0.0

    # Create set of true edges
    if isinstance(gold_standard, set):
        true_edges = gold_standard
    else:  # pd.DataFrame
        true_edges = set(zip(gold_standard["tf"], gold_standard["target"]))

    # Sort edges by score (descending) and take top k
    edge_list.sort(key=lambda x: x[2], reverse=True)
    top_k_edges = edge_list[:k]

    # Count true positives in top k
    true_positives = sum(
        1 for tf, target, _ in top_k_edges if (tf, target) in true_edges
    )

    precision = true_positives / k if k > 0 else 0.0

    return float(precision)


def compute_recall_at_k(
    inferred_edges: dict[tuple[str, str], float] | "nx.DiGraph",
    gold_standard: "pd.DataFrame" | set[tuple[str, str]],
    k: int = 100,
) -> float:
    """Compute recall of top-k predicted edges.

    Recall@k measures the fraction of true edges that are recovered
    in the top-k highest-scoring predictions.

    Parameters
    ----------
    inferred_edges : dict or nx.DiGraph
        Inferred regulatory edges with scores

    gold_standard : pd.DataFrame or set
        Gold standard regulatory network

    k : int, default=100
        Number of top predictions to consider

    Returns
    -------
    recall_at_k : float
        Recall@k = (true positives in top k) / (total true edges)
        Ranges from 0 to 1, higher is better.

    Examples
    --------
    >>> inferred = {('TF1', 'GENE1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> gold_standard = {('TF1', 'GENE1'), ('TF2', 'GENE3')}
    >>> recall = compute_recall_at_k(inferred, gold_standard, k=10)
    """
    # Convert inputs to consistent format
    if isinstance(inferred_edges, dict):
        # Dict items are ((tf, target), score) tuples
        edge_list = [(tf, target, score) for (tf, target), score in inferred_edges.items()]
    else:  # nx.DiGraph
        edge_list = [
            (u, v, data.get("weight", 1.0))
            for u, v, data in inferred_edges.edges(data=True)
        ]

    if not edge_list:
        return 0.0

    # Create set of true edges
    if isinstance(gold_standard, set):
        true_edges = gold_standard
    else:  # pd.DataFrame
        true_edges = set(zip(gold_standard["tf"], gold_standard["target"]))

    if not true_edges:
        return 0.0

    # Sort edges by score (descending) and take top k
    edge_list.sort(key=lambda x: x[2], reverse=True)
    top_k_edges = edge_list[:k]

    # Count true positives in top k
    true_positives = sum(
        1 for tf, target, _ in top_k_edges if (tf, target) in true_edges
    )

    recall = true_positives / len(true_edges)

    return float(recall)


def compute_f1_at_k(
    inferred_edges: dict[tuple[str, str], float] | "nx.DiGraph",
    gold_standard: "pd.DataFrame" | set[tuple[str, str]],
    k: int = 100,
) -> float:
    """Compute F1 score of top-k predicted edges.

    F1@k is the harmonic mean of precision@k and recall@k.

    Parameters
    ----------
    inferred_edges : dict or nx.DiGraph
        Inferred regulatory edges with scores

    gold_standard : pd.DataFrame or set
        Gold standard regulatory network

    k : int, default=100
        Number of top predictions to consider

    Returns
    -------
    f1_at_k : float
        F1@k = 2 * (precision@k * recall@k) / (precision@k + recall@k)
        Ranges from 0 to 1, higher is better.

    Examples
    --------
    >>> inferred = {('TF1', 'GENE1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> gold_standard = {('TF1', 'GENE1')}
    >>> f1 = compute_f1_at_k(inferred, gold_standard, k=10)
    """
    precision = compute_precision_at_k(inferred_edges, gold_standard, k)
    recall = compute_recall_at_k(inferred_edges, gold_standard, k)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)

    return float(f1)


def evaluate_grn(
    inferred_edges: dict[tuple[str, str], float] | "nx.DiGraph",
    gold_standard: "pd.DataFrame" | set[tuple[str, str]],
    *,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute all standard GRN evaluation metrics.

    Parameters
    ----------
    inferred_edges : dict or nx.DiGraph
        Inferred regulatory edges with scores

    gold_standard : pd.DataFrame or set
        Gold standard regulatory network

    k_values : list of int, optional
        Values of k for precision@k, recall@k, f1@k.
        Default is [100, 500, 1000]

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics:
        - 'auroc': Area Under ROC Curve
        - 'aupr': Area Under Precision-Recall Curve
        - 'precision@{k}': Precision at k for each k in k_values
        - 'recall@{k}': Recall at k for each k in k_values
        - 'f1@{k}': F1 score at k for each k in k_values

    Examples
    --------
    >>> inferred = {('TF1', 'GENE1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> gold_standard = {('TF1', 'GENE1')}
    >>> metrics = evaluate_grn(inferred, gold_standard)
    >>> print(f"AUPR: {metrics['aupr']:.3f}")
    """
    if k_values is None:
        k_values = [100, 500, 1000]

    metrics = {}

    # Primary metrics
    metrics["auroc"] = compute_auroc(inferred_edges, gold_standard)
    metrics["aupr"] = compute_aupr(inferred_edges, gold_standard)

    # Precision@k, Recall@k, F1@k for different k values
    for k in k_values:
        metrics[f"precision@{k}"] = compute_precision_at_k(
            inferred_edges, gold_standard, k
        )
        metrics[f"recall@{k}"] = compute_recall_at_k(inferred_edges, gold_standard, k)
        metrics[f"f1@{k}"] = compute_f1_at_k(inferred_edges, gold_standard, k)

    return metrics
