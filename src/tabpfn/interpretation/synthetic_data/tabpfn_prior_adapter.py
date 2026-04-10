"""Adapter for the external TabPFN synthetic data generator.

Bridges `zzhang-cn/tabpfn-synthetic-data` (faithful reimplementation of
TabPFN's training prior) with our interpretation model's label generator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from tabpfn.interpretation.synthetic_data.scm_generator import SCMDataset

logger = logging.getLogger(__name__)

# Default config path for v2.5-extended prior
_DEFAULT_CONFIG = Path(__file__).parent / "configs" / "v2_5_prior.yaml"


class TabPFNPriorAdapter:
    """Generate datasets using the external TabPFN prior reimplementation.

    Wraps `SyntheticDataGenerator` from `zzhang-cn/tabpfn-synthetic-data`,
    extracts the causal graph from metadata, and returns ``SCMDataset``
    objects compatible with our label generator.

    Parameters
    ----------
    config_path : str or Path, optional
        YAML configuration file. Defaults to the bundled v2.5 prior config.
    seed : int, optional
        Random seed for reproducibility.
    n_test : int
        Fixed number of test samples (matching TabPFN's prior). Default 128.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        seed: int | None = None,
        n_test: int = 128,
    ) -> None:
        try:
            from src.data_generation import SyntheticDataGenerator
        except ImportError as e:
            raise ImportError(
                "tabpfn-synthetic-data is required. Install with:\n"
                "  pip install git+https://github.com/zzhang-cn/tabpfn-synthetic-data.git"
            ) from e

        if config_path is None:
            config_path = str(_DEFAULT_CONFIG)
        self._config_path = str(config_path)
        self._seed = seed
        self.n_test = n_test
        self._generator = SyntheticDataGenerator(
            config_path=self._config_path, seed=seed
        )
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        *,
        n_features: int | None = None,
        n_samples: int | None = None,
    ) -> SCMDataset:
        """Generate a single dataset with known causal structure.

        Parameters
        ----------
        n_features : int, optional
            Number of features. Sampled from prior if *None*.
        n_samples : int, optional
            Total number of samples (train + test). If *None*, sampled from
            prior then adjusted so that ``n_test`` samples are reserved.

        Returns
        -------
        SCMDataset
            Dataset with ``X``, ``y``, ``dag``, and metadata.
        """
        max_retries = 10
        for attempt in range(max_retries):
            result = self._try_generate(n_features=n_features, n_samples=n_samples)
            if result is not None:
                # Check that target has at least 1 parent
                if result.dag.in_degree(result.target_node) >= 1:
                    return result
                logger.debug(
                    "Attempt %d: target has 0 parents, retrying", attempt + 1
                )
        # Final attempt — return even without parents
        result = self._try_generate(n_features=n_features, n_samples=n_samples)
        if result is not None:
            return result
        raise RuntimeError("Failed to generate dataset after retries")

    def _try_generate(
        self,
        *,
        n_features: int | None = None,
        n_samples: int | None = None,
    ) -> SCMDataset | None:
        """Single generation attempt. Returns None on failure."""
        try:
            return self._generate_inner(
                n_features=n_features, n_samples=n_samples
            )
        except Exception as exc:
            logger.debug("Generation attempt failed: %s", exc)
            return None

    def _generate_inner(
        self,
        *,
        n_features: int | None = None,
        n_samples: int | None = None,
    ) -> SCMDataset:
        """Core generation logic."""
        # Determine total samples: n_train + n_test
        if n_samples is not None:
            total = n_samples
        else:
            total = int(self._rng.integers(50, 10001)) + self.n_test

        dataset, metadata = self._generator.generate_dataset(
            n_samples=total,
            n_features=n_features,
            task_type="regression",
            return_metadata=True,
        )

        X_train, y_train = dataset["train"]
        X_test, y_test = dataset["test"]

        # Stack full dataset
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])

        # Handle NaN from missing-value post-processing
        nan_mask = np.isnan(X)
        if nan_mask.any():
            col_means = np.nanmean(X, axis=0)
            col_means = np.where(np.isnan(col_means), 0.0, col_means)
            for col_idx in range(X.shape[1]):
                X[nan_mask[:, col_idx], col_idx] = col_means[col_idx]
        if np.isnan(y).any():
            y = np.where(np.isnan(y), np.nanmean(y), y)

        # Reconstruct the DAG
        dag, feature_to_node, target_node_id = self._reconstruct_dag(metadata)

        # Build a *feature-level* DAG where:
        #   node i (0..n_features-1) = feature i
        #   node n_features = target
        n_feat = X.shape[1]
        feature_dag = self._build_feature_dag(
            dag, feature_to_node, target_node_id, n_feat
        )

        feature_names = [f"X_{i}" for i in range(n_feat)]

        md: dict[str, Any] = {
            "n_features": n_feat,
            "n_samples": X.shape[0],
            "n_train": X_train.shape[0],
            "n_test": X_test.shape[0],
            "graph_type": "scale_free",
            "source": "tabpfn_prior",
            "node_dag_nodes": dag.number_of_nodes(),
            "node_dag_edges": dag.number_of_edges(),
            "n_edges": feature_dag.number_of_edges(),
            "n_target_parents": feature_dag.in_degree(n_feat),
            "feature_to_node": feature_to_node,
            "target_node_id": target_node_id,
            "edge_types": metadata.get("edge_types", {}),
        }

        return SCMDataset(
            X=X,
            y=y,
            dag=feature_dag,
            target_node=n_feat,
            feature_names=feature_names,
            edge_functions={},  # Not needed for label generation
            noise_std=0.0,
            metadata=md,
        )

    def generate_batch(self, n_datasets: int, **kwargs: Any) -> list[SCMDataset]:
        """Generate a batch of datasets."""
        results: list[SCMDataset] = []
        for i in range(n_datasets):
            try:
                ds = self.generate(**kwargs)
                results.append(ds)
            except Exception as exc:
                logger.warning("Dataset %d failed: %s", i, exc)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_dag(metadata: dict) -> tuple[nx.DiGraph, dict[int, int], int]:
        """Reconstruct the node-level DAG and feature→node mapping.

        Returns
        -------
        dag : nx.DiGraph
            Node-level directed acyclic graph.
        feature_to_node : dict[int, int]
            Maps feature index → node_id in the DAG.
        target_node_id : int
            The node_id of the target variable.
        """
        # Build node-level DAG from edge_types
        dag = nx.DiGraph()
        edge_types = metadata.get("edge_types", {})
        for edge_str in edge_types:
            # edge_str looks like "(1, 0)"
            parts = edge_str.strip("()").split(",")
            u, v = int(parts[0].strip()), int(parts[1].strip())
            dag.add_edge(u, v)

        # Feature → node mapping from feature_metadata
        feature_metadata = metadata.get("feature_metadata", [])
        feature_to_node: dict[int, int] = {}
        for i, fm in enumerate(feature_metadata):
            feature_to_node[i] = fm["node_id"]

        # Target node
        target_meta = metadata.get("target_metadata", {})
        target_node_id = target_meta.get("node_id", -1)

        # Ensure all referenced nodes exist
        all_nodes = set(dag.nodes())
        for nid in feature_to_node.values():
            all_nodes.add(nid)
        all_nodes.add(target_node_id)
        dag.add_nodes_from(all_nodes)

        return dag, feature_to_node, target_node_id

    @staticmethod
    def _build_feature_dag(
        node_dag: nx.DiGraph,
        feature_to_node: dict[int, int],
        target_node_id: int,
        n_features: int,
    ) -> nx.DiGraph:
        """Build a feature-level DAG from the node-level DAG.

        Only adds ``feature_i → target`` when the feature's source node is
        a **direct parent** of the target node.  Indirect ancestry is
        captured through inter-feature edges (``feature_i → feature_j``
        when feature_i's node has a directed path to feature_j's node).
        """
        feat_dag = nx.DiGraph()
        feat_dag.add_nodes_from(range(n_features + 1))  # features + target

        target_idx = n_features

        # Direct parents of target in the node-level DAG
        direct_parent_nodes = set(node_dag.predecessors(target_node_id))

        # Feature → target: only for DIRECT parents
        for feat_idx in range(n_features):
            src_node = feature_to_node.get(feat_idx)
            if src_node is None or src_node == target_node_id:
                continue
            if src_node in direct_parent_nodes:
                feat_dag.add_edge(feat_idx, target_idx)

        # Inter-feature edges: fi → fj if fi's node has a direct edge
        # to fj's node in the node-level DAG
        for fi in range(n_features):
            ni = feature_to_node.get(fi)
            if ni is None:
                continue
            for fj in range(n_features):
                if fi == fj:
                    continue
                nj = feature_to_node.get(fj)
                if nj is None or ni == nj:
                    continue
                if node_dag.has_edge(ni, nj):
                    feat_dag.add_edge(fi, fj)

        return feat_dag
