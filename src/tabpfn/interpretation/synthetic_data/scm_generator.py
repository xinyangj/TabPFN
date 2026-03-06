"""SCM-based synthetic data generator.

Generates tabular datasets from random Structural Causal Models (SCMs)
with known ground-truth causal structure, suitable for training an
interpretation model that predicts feature importance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import networkx as nx
import numpy as np

from tabpfn.interpretation.synthetic_data.dag_generator import (
    ensure_target_has_parents,
    generate_random_dag,
)
from tabpfn.interpretation.synthetic_data.edge_functions import (
    EdgeFunction,
    sample_edge_function,
)


@dataclass
class SCMDataset:
    """A synthetic dataset generated from an SCM with known causal structure.

    Attributes
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target vector of shape (n_samples,).
    dag : nx.DiGraph
        The causal DAG used to generate the data. Node indices 0..n_features-1
        correspond to features, and node n_features is the target.
    target_node : int
        The index of the target node in the DAG.
    feature_names : list[str]
        Names of the features (X_0, X_1, ...).
    edge_functions : dict[tuple[int, int], EdgeFunction]
        The edge functions used for each edge in the DAG.
    noise_std : float
        Standard deviation of additive noise at each node.
    metadata : dict[str, Any]
        Additional metadata about the generation process.
    """

    X: np.ndarray
    y: np.ndarray
    dag: nx.DiGraph
    target_node: int
    feature_names: list[str]
    edge_functions: dict[tuple[int, int], EdgeFunction]
    noise_std: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SCMGenerator:
    """Generate synthetic datasets from random Structural Causal Models.

    Each generated dataset has a known causal DAG, allowing extraction
    of ground-truth feature importance labels.

    Parameters
    ----------
    n_features_range : tuple[int, int]
        Range of number of features (min, max) to sample from.
    n_samples_range : tuple[int, int]
        Range of number of samples (min, max) to sample from.
    noise_std_range : tuple[float, float]
        Range of noise standard deviation at each node.
    edge_prob_range : tuple[float, float]
        Range of edge probability for DAG generation.
    graph_type : str
        Type of random graph to generate.
    seed : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> generator = SCMGenerator(seed=42)
    >>> dataset = generator.generate()
    >>> print(f"X shape: {dataset.X.shape}, target parents: "
    ...       f"{list(dataset.dag.predecessors(dataset.target_node))}")
    """

    def __init__(
        self,
        *,
        n_features_range: tuple[int, int] = (5, 30),
        n_samples_range: tuple[int, int] = (100, 500),
        noise_std_range: tuple[float, float] = (0.1, 1.0),
        edge_prob_range: tuple[float, float] = (0.15, 0.5),
        graph_type: Literal[
            "erdos_renyi", "scale_free", "chain", "tree"
        ] = "erdos_renyi",
        seed: int | None = None,
    ) -> None:
        self.n_features_range = n_features_range
        self.n_samples_range = n_samples_range
        self.noise_std_range = noise_std_range
        self.edge_prob_range = edge_prob_range
        self.graph_type = graph_type
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        *,
        n_features: int | None = None,
        n_samples: int | None = None,
        noise_std: float | None = None,
        edge_prob: float | None = None,
    ) -> SCMDataset:
        """Generate a single synthetic dataset from a random SCM.

        Parameters
        ----------
        n_features : int, optional
            Number of features. If None, sampled from n_features_range.
        n_samples : int, optional
            Number of samples. If None, sampled from n_samples_range.
        noise_std : float, optional
            Noise standard deviation. If None, sampled from noise_std_range.
        edge_prob : float, optional
            Edge probability. If None, sampled from edge_prob_range.

        Returns
        -------
        SCMDataset
            The generated dataset with known causal structure.
        """
        if n_features is None:
            n_features = int(
                self.rng.integers(self.n_features_range[0], self.n_features_range[1] + 1)
            )
        if n_samples is None:
            n_samples = int(
                self.rng.integers(self.n_samples_range[0], self.n_samples_range[1] + 1)
            )
        if noise_std is None:
            noise_std = float(
                self.rng.uniform(self.noise_std_range[0], self.noise_std_range[1])
            )
        if edge_prob is None:
            edge_prob = float(
                self.rng.uniform(self.edge_prob_range[0], self.edge_prob_range[1])
            )

        n_nodes = n_features + 1  # features + target
        target_node = n_features  # last node is target

        # Generate DAG
        dag = generate_random_dag(
            n_nodes, graph_type=self.graph_type, edge_prob=edge_prob, rng=self.rng
        )
        # Ensure target has at least 1 parent
        dag = ensure_target_has_parents(dag, target_node, min_parents=1, rng=self.rng)

        # Assign edge functions
        edge_functions: dict[tuple[int, int], EdgeFunction] = {}
        for u, v in dag.edges():
            edge_functions[(u, v)] = sample_edge_function(self.rng)

        # Generate data by propagating through DAG in topological order
        node_values = np.zeros((n_samples, n_nodes))
        topo_order = list(nx.topological_sort(dag))

        for node in topo_order:
            parents = list(dag.predecessors(node))
            if len(parents) == 0:
                # Root node: sample from standard normal
                node_values[:, node] = self.rng.standard_normal(n_samples)
            else:
                # Compute value from parents
                value = np.zeros(n_samples)
                for parent in parents:
                    edge_fn = edge_functions[(parent, node)]
                    value += edge_fn(node_values[:, parent])
                # Add noise
                value += self.rng.normal(0, noise_std, size=n_samples)
                node_values[:, node] = value

        # Split into X (features) and y (target)
        X = node_values[:, :n_features]
        y = node_values[:, target_node]

        feature_names = [f"X_{i}" for i in range(n_features)]

        metadata = {
            "n_features": n_features,
            "n_samples": n_samples,
            "noise_std": noise_std,
            "edge_prob": edge_prob,
            "graph_type": self.graph_type,
            "n_edges": dag.number_of_edges(),
            "n_target_parents": dag.in_degree(target_node),
        }

        return SCMDataset(
            X=X,
            y=y,
            dag=dag,
            target_node=target_node,
            feature_names=feature_names,
            edge_functions=edge_functions,
            noise_std=noise_std,
            metadata=metadata,
        )

    def generate_batch(
        self,
        n_datasets: int,
        **kwargs: Any,
    ) -> list[SCMDataset]:
        """Generate a batch of synthetic datasets.

        Parameters
        ----------
        n_datasets : int
            Number of datasets to generate.
        **kwargs
            Additional arguments passed to generate().

        Returns
        -------
        list[SCMDataset]
            List of generated datasets.
        """
        return [self.generate(**kwargs) for _ in range(n_datasets)]
