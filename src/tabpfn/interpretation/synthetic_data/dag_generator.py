"""Random DAG generation for Structural Causal Models.

Generates random directed acyclic graphs (DAGs) with configurable topology
for use in synthetic data generation with known causal structure.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np


def generate_random_dag(
    n_nodes: int,
    *,
    graph_type: Literal["erdos_renyi", "scale_free", "chain", "tree"] = "erdos_renyi",
    edge_prob: float = 0.3,
    rng: np.random.Generator | None = None,
) -> nx.DiGraph:
    """Generate a random DAG with specified topology.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    graph_type : str
        Type of random graph to generate.
    edge_prob : float
        Edge probability (for erdos_renyi) or density parameter.
    rng : np.random.Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    nx.DiGraph
        A directed acyclic graph with nodes labeled 0..n_nodes-1.
    """
    if rng is None:
        rng = np.random.default_rng()

    if graph_type == "erdos_renyi":
        return _erdos_renyi_dag(n_nodes, edge_prob, rng)
    elif graph_type == "scale_free":
        return _scale_free_dag(n_nodes, rng)
    elif graph_type == "chain":
        return _chain_dag(n_nodes, rng)
    elif graph_type == "tree":
        return _tree_dag(n_nodes, rng)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")


def _erdos_renyi_dag(
    n_nodes: int, edge_prob: float, rng: np.random.Generator
) -> nx.DiGraph:
    """Generate an Erdős-Rényi random DAG.

    Uses a random permutation to define topological order, then adds
    edges from lower to higher order with probability edge_prob.
    """
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_nodes))

    # Random topological ordering
    perm = rng.permutation(n_nodes)

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() < edge_prob:
                dag.add_edge(int(perm[i]), int(perm[j]))

    return dag


def _scale_free_dag(n_nodes: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a scale-free DAG using preferential attachment.

    Nodes are added sequentially, each connecting to 1-3 existing nodes
    with probability proportional to in-degree + 1.
    """
    dag = nx.DiGraph()
    dag.add_node(0)

    for node in range(1, n_nodes):
        dag.add_node(node)
        existing = list(range(node))
        # Preferential attachment weights
        weights = np.array([dag.in_degree(n) + 1.0 for n in existing])
        weights /= weights.sum()

        n_parents = min(rng.integers(1, 4), len(existing))
        parents = rng.choice(existing, size=n_parents, replace=False, p=weights)
        for parent in parents:
            dag.add_edge(int(parent), node)

    return dag


def _chain_dag(n_nodes: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a simple chain DAG with random ordering."""
    dag = nx.DiGraph()
    perm = rng.permutation(n_nodes)
    dag.add_nodes_from(range(n_nodes))
    for i in range(n_nodes - 1):
        dag.add_edge(int(perm[i]), int(perm[i + 1]))
    return dag


def _tree_dag(n_nodes: int, rng: np.random.Generator) -> nx.DiGraph:
    """Generate a random tree DAG."""
    dag = nx.DiGraph()
    dag.add_node(0)
    for node in range(1, n_nodes):
        parent = rng.integers(0, node)
        dag.add_edge(int(parent), node)
    return dag


def ensure_target_has_parents(
    dag: nx.DiGraph,
    target_node: int,
    *,
    min_parents: int = 1,
    rng: np.random.Generator | None = None,
) -> nx.DiGraph:
    """Ensure the target node has at least min_parents direct parents.

    If the target has fewer parents, randomly add edges from other nodes
    that won't create cycles.

    Parameters
    ----------
    dag : nx.DiGraph
        The DAG to modify (modified in place).
    target_node : int
        The target node that must have parents.
    min_parents : int
        Minimum number of parents the target should have.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    nx.DiGraph
        The modified DAG.
    """
    if rng is None:
        rng = np.random.default_rng()

    current_parents = list(dag.predecessors(target_node))
    if len(current_parents) >= min_parents:
        return dag

    # Find nodes that can be parents without creating cycles
    descendants = nx.descendants(dag, target_node)
    candidates = [
        n
        for n in dag.nodes()
        if n != target_node and n not in current_parents and n not in descendants
    ]

    n_needed = min_parents - len(current_parents)
    if len(candidates) < n_needed:
        n_needed = len(candidates)

    if n_needed > 0:
        new_parents = rng.choice(candidates, size=n_needed, replace=False)
        for parent in new_parents:
            dag.add_edge(int(parent), target_node)

    return dag
