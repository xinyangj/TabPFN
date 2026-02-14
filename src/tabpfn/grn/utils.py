"""Utility functions for GRN inference and evaluation.

This module provides reusable utilities for:
- Filtering self-edges from edge scores
- Computing feature indices for targets (with information leakage prevention)
- Creating edge score dictionaries
- Evaluating and formatting results with consistent output

These utilities eliminate code duplication across baseline methods and ensure
consistent handling of the critical information leakage prevention fix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from tabpfn.grn.evaluation import GoldStandardType


def filter_self_edges(
    edge_scores: dict[tuple[str, str], float]
) -> dict[tuple[str, str], float]:
    """Filter out self-edges (TF regulating itself) from edge scores.

    Self-edges occur when TF == target, which can happen in GRN inference
    when allow_overlap=True (targets are also TFs). These edges are biologically
    meaningless and should be filtered out.

    Parameters
    ----------
    edge_scores : dict[tuple[str, str], float]
        Dictionary mapping (tf, target) pairs to edge scores

    Returns
    -------
    dict[tuple[str, str], float]
        Dictionary with self-edges removed (where tf == target)

    Examples
    --------
    >>> scores = {('TF1', 'TF1'): 0.9, ('TF1', 'GENE2'): 0.3}
    >>> filter_self_edges(scores)
    {('TF1', 'GENE2'): 0.3}
    """
    return {(tf, tgt): score for (tf, tgt), score in edge_scores.items() if tf != tgt}


def compute_target_feature_indices(
    tf_names: list[str],
    target_name: str | None,
    warn_on_single: bool = True
) -> tuple[list[int], list[str], dict[int, int]]:
    """Compute feature indices for a target, excluding self if needed.

    CRITICAL: This implements the information leakage prevention logic.
    When a target gene is also a TF (target_name in tf_names), it must be
    excluded from the input features during training to prevent the model from
    simply copying the target's value instead of learning regulatory relationships.

    Uses name-based comparison (consistent with grn_regressor.py), not index-based.

    Parameters
    ----------
    tf_names : list[str]
        List of all TF names (potential input features)
    target_name : str | None
        Name of target gene. If None, uses all TFs as features.
    warn_on_single : bool, default=True
        Whether to warn when target is the only TF (edge case)

    Returns
    -------
    tf_indices : list[int]
        Indices of TFs to use as input features
    tf_names_for_target : list[str]
        Names of TFs to use as input features
    modified_to_original_idx_map : dict[int, int]
        Maps feature index (0 to n_features-1) to original TF index.
        Used to convert attention/feature indices back to original TF indices.

    Examples
    --------
    >>> tf_names = ['TF1', 'TF2', 'TF3']
    >>> target_name = 'TF2'
    >>> indices, names, idx_map = compute_target_feature_indices(tf_names, target_name)
    >>> indices
    [0, 2]
    >>> names
    ['TF1', 'TF3']
    >>> idx_map
    {0: 0, 1: 2}
    """
    tf_indices_excluding_target = []
    tf_names_excluding_target = []
    modified_to_original_idx_map = {}
    modified_idx = 0

    for orig_idx, tf_name in enumerate(tf_names):
        if tf_name == target_name:
            continue  # Skip the target gene itself
        tf_indices_excluding_target.append(orig_idx)
        tf_names_excluding_target.append(tf_name)
        # CRITICAL FIX: Only increment modified_idx for INCLUDED TFs
        # NOT when skipping (target is excluded)
        modified_to_original_idx_map[modified_idx] = orig_idx
        modified_idx += 1

    # Edge case handling: target is the only TF
    if len(tf_indices_excluding_target) == 0:
        if warn_on_single and target_name is not None:
            import warnings
            warnings.warn(
                f"Target '{target_name}' is the only TF. "
                "Using all features (potential information leakage)."
            )
        # Use all features with warning
        return (
            list(range(len(tf_names))),
            tf_names.copy(),
            {i: i for i in range(len(tf_names))},
        )

    return tf_indices_excluding_target, tf_names_excluding_target, modified_to_original_idx_map


def compute_feature_to_block_mapping(
    n_features: int,
    n_feat_pos: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Map raw feature indices to attention block indices.

    TabPFN encodes each raw feature as 2 values (value + nan_indicator),
    then groups every ``features_per_group=3`` encoded values into one
    attention block.  The target is appended as the last block.

    Parameters
    ----------
    n_features : int
        Number of raw input features (TFs).
    n_feat_pos : int
        Number of attention positions (from actual attention tensor shape).
        The last position is the target block.

    Returns
    -------
    feature_to_blocks : dict[int, list[int]]
        Maps each raw feature index to its attention block index(es).
    block_to_features : dict[int, list[int]]
        Maps each attention block index to its raw feature index(es).
    """
    n_blocks = n_feat_pos - 1  # exclude target position
    feature_to_blocks: dict[int, list[int]] = {}
    block_to_features: dict[int, list[int]] = {b: [] for b in range(n_blocks)}

    for i in range(n_features):
        b0 = (2 * i) // 3
        b1 = (2 * i + 1) // 3
        blocks = sorted({min(b0, n_blocks - 1), min(b1, n_blocks - 1)})
        feature_to_blocks[i] = blocks
        for b in blocks:
            if i not in block_to_features[b]:
                block_to_features[b].append(i)

    return feature_to_blocks, block_to_features


def create_edge_score_dict(
    tf_names: list[str],
    target_genes: list[str],
    scores: np.ndarray | dict,
    skip_self_edges: bool = True
) -> dict[tuple[str, str], float]:
    """Create edge score dictionary from TF-target scores.

    Parameters
    ----------
    tf_names : list[str]
        List of TF names (rows of score matrix)
    target_genes : list[str]
        List of target gene names (columns of score matrix)
    scores : np.ndarray or dict
        Score matrix of shape (n_tfs, n_targets) or pre-built dict
    skip_self_edges : bool, default=True
        Whether to exclude self-edges where tf == target

    Returns
    -------
    dict[tuple[str, str], float]
        Dictionary mapping (tf, target) pairs to edge scores

    Examples
    --------
    >>> tf_names = ['TF1', 'TF2']
    >>> targets = ['G1', 'G2']
    >>> scores = np.array([[0.1, 0.9], [0.8, 0.2]])
    >>> create_edge_score_dict(tf_names, targets, scores)
    {('TF1', 'G1'): 0.1, ('TF1', 'G2'): 0.9, ('TF2', 'G1'): 0.8, ('TF2', 'G2'): 0.2}
    """
    edge_scores: dict[tuple[str, str], float] = {}

    if isinstance(scores, dict):
        # Already a dict, just filter self-edges if needed
        edge_scores = dict(scores)
        if skip_self_edges:
            edge_scores = filter_self_edges(edge_scores)
        return edge_scores

    # Build dict from numpy array
    for i, tf_name in enumerate(tf_names):
        for j, target_name in enumerate(target_genes):
            if skip_self_edges and tf_name == target_name:
                continue  # Skip self-edges
            edge_scores[(tf_name, target_name)] = float(scores[i, j])

    return edge_scores


def evaluate_and_format_results(
    edge_scores: dict[tuple[str, str], float],
    gold_standard: "GoldStandardType",
    dataset_name: str,
    method_name: str,
    training_time: float = 0.0,
) -> dict:
    """Evaluate GRN predictions and format results with printing.

    This function combines the common pattern of:
    1. Computing k_values based on gold_standard size
    2. Calling evaluate_grn()
    3. Augmenting metrics with training time and edge counts
    4. Printing results
    5. Returning formatted results dict

    This eliminates ~68 lines of duplication across baseline functions.

    Parameters
    ----------
    edge_scores : dict[tuple[str, str], float]
        Predicted edge scores from the GRN inference method
    gold_standard : GoldStandardType
        True regulatory edges (set of tuples or DataFrame)
    dataset_name : str
        Name of dataset (for printing)
    method_name : str
        Name of method (for printing)
    training_time : float, default=0.0
        Training time in seconds

    Returns
    -------
    dict
        Dictionary with keys: dataset, method, metrics, edge_scores
        - metrics contains: aupr, auroc, precision@k, training_time, etc.
    """
    from tabpfn.grn import evaluate_grn

    # Print header
    print(f"\n{'='*70}")
    print(f"{method_name} Results: {dataset_name}")
    print(f"{'='*70}")

    # Compute k_values based on gold_standard size
    if isinstance(gold_standard, set):
        num_edges = len(gold_standard)
    else:  # DataFrame
        num_edges = len(gold_standard)
    k_values = [10, 50, 100] if num_edges > 50 else [5, 10, 20]

    # Evaluate GRN predictions
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

    # Augment metrics with additional info
    metrics["training_time"] = training_time
    metrics["num_edges_predicted"] = len(edge_scores)
    metrics["num_true_edges"] = num_edges

    # Print results
    print(f"\nResults:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    if training_time > 0:
        print(f"  Time:   {training_time:.2f}s")

    # Return formatted results dict
    return {
        "dataset": dataset_name,
        "method": method_name,
        "metrics": metrics,
        "edge_scores": edge_scores,
    }
