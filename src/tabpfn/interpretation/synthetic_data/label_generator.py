"""Ground-truth feature importance labels from SCM DAGs.

Provides 4 modes of labeling feature importance:
1. Binary direct parents
2. Binary ancestry (all ancestors)
3. Graded ancestry with path decay
4. Interventional sensitivity
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from tabpfn.interpretation.synthetic_data.scm_generator import SCMDataset

LabelMode = Literal[
    "binary_direct", "binary_ancestry", "graded_ancestry", "interventional"
]


def compute_importance_labels(
    dataset: SCMDataset,
    mode: LabelMode = "binary_direct",
    *,
    decay_factor: float = 0.5,
    n_intervention_samples: int = 200,
) -> np.ndarray:
    """Compute ground-truth feature importance labels from an SCM dataset.

    Parameters
    ----------
    dataset : SCMDataset
        The synthetic dataset with known causal DAG.
    mode : LabelMode
        Which labeling mode to use:
        - "binary_direct": 1.0 for direct parents of target, 0.0 otherwise
        - "binary_ancestry": 1.0 for all ancestors (direct + indirect), 0.0 otherwise
        - "graded_ancestry": decay^path_length for ancestors, 0.0 for non-ancestors
        - "interventional": Causal effect magnitude via do-calculus simulation
    decay_factor : float
        Decay factor for graded_ancestry mode. Default 0.5.
    n_intervention_samples : int
        Number of samples for interventional mode. Default 200.

    Returns
    -------
    np.ndarray
        Feature importance vector of shape (n_features,), values in [0, 1].
    """
    dag = dataset.dag
    target = dataset.target_node
    n_features = dataset.X.shape[1]

    if mode == "binary_direct":
        return _binary_direct(dag, target, n_features)
    elif mode == "binary_ancestry":
        return _binary_ancestry(dag, target, n_features)
    elif mode == "graded_ancestry":
        return _graded_ancestry(dag, target, n_features, decay_factor)
    elif mode == "interventional":
        return _interventional(dataset, n_intervention_samples)
    else:
        raise ValueError(f"Unknown label mode: {mode}")


def compute_all_labels(
    dataset: SCMDataset,
    *,
    decay_factor: float = 0.5,
    n_intervention_samples: int = 200,
) -> dict[LabelMode, np.ndarray]:
    """Compute all 4 modes of importance labels for a dataset.

    Returns
    -------
    dict[LabelMode, np.ndarray]
        Dictionary mapping mode name to importance vector.
    """
    return {
        "binary_direct": compute_importance_labels(dataset, "binary_direct"),
        "binary_ancestry": compute_importance_labels(dataset, "binary_ancestry"),
        "graded_ancestry": compute_importance_labels(
            dataset, "graded_ancestry", decay_factor=decay_factor
        ),
        "interventional": compute_importance_labels(
            dataset,
            "interventional",
            n_intervention_samples=n_intervention_samples,
        ),
    }


def _binary_direct(dag: nx.DiGraph, target: int, n_features: int) -> np.ndarray:
    """Mode 1: Binary indicator for direct parents of target."""
    importance = np.zeros(n_features)
    for parent in dag.predecessors(target):
        if parent < n_features:
            importance[parent] = 1.0
    return importance


def _binary_ancestry(dag: nx.DiGraph, target: int, n_features: int) -> np.ndarray:
    """Mode 2: Binary indicator for ALL ancestors of target (direct + indirect)."""
    importance = np.zeros(n_features)
    ancestors = nx.ancestors(dag, target)
    for ancestor in ancestors:
        if ancestor < n_features:
            importance[ancestor] = 1.0
    return importance


def _graded_ancestry(
    dag: nx.DiGraph, target: int, n_features: int, decay: float
) -> np.ndarray:
    """Mode 3: Graded importance based on shortest path distance with decay.

    For each ancestor, importance = decay^(shortest_path_length).
    Takes the maximum importance across all paths.
    Normalized so direct parents have importance 1.0.
    """
    importance = np.zeros(n_features)
    ancestors = nx.ancestors(dag, target)

    for ancestor in ancestors:
        if ancestor < n_features:
            try:
                # Shortest path from ancestor to target
                path_length = nx.shortest_path_length(dag, ancestor, target)
                importance[ancestor] = decay ** (path_length - 1)
            except nx.NetworkXNoPath:
                pass

    return importance


def _interventional(
    dataset: SCMDataset,
    n_samples: int,
) -> np.ndarray:
    """Mode 4: Interventional sensitivity via do-calculus simulation.

    For each feature X_i, compute |E[Y | do(X_i = high)] - E[Y | do(X_i = low)]|
    by simulating interventions through the SCM.
    """
    dag = dataset.dag
    target = dataset.target_node
    n_features = dataset.X.shape[1]
    n_nodes = n_features + 1

    importance = np.zeros(n_features)
    topo_order = list(nx.topological_sort(dag))

    # Use dataset's RNG seed for reproducibility (use a fixed seed)
    rng = np.random.default_rng(42)

    for feature_idx in range(n_features):
        # Simulate intervention: do(X_i = high) vs do(X_i = low)
        x_low = np.percentile(dataset.X[:, feature_idx], 10)
        x_high = np.percentile(dataset.X[:, feature_idx], 90)

        y_values = np.zeros((2, n_samples))  # [low_intervention, high_intervention]

        for interv_idx, interv_value in enumerate([x_low, x_high]):
            node_values = np.zeros((n_samples, n_nodes))
            # Sample root node values
            for node in topo_order:
                parents = list(dag.predecessors(node))
                if node == feature_idx:
                    # Intervention: set to fixed value
                    node_values[:, node] = interv_value
                elif len(parents) == 0:
                    node_values[:, node] = rng.standard_normal(n_samples)
                else:
                    value = np.zeros(n_samples)
                    for parent in parents:
                        edge_fn = dataset.edge_functions[(parent, node)]
                        value += edge_fn(node_values[:, parent])
                    value += rng.normal(0, dataset.noise_std, size=n_samples)
                    node_values[:, node] = value

            y_values[interv_idx] = node_values[:, target]

        # Causal effect = |E[Y|do(X_i=high)] - E[Y|do(X_i=low)]|
        importance[feature_idx] = abs(y_values[1].mean() - y_values[0].mean())

    # Normalize to [0, 1]
    max_imp = importance.max()
    if max_imp > 0:
        importance = importance / max_imp

    return importance
