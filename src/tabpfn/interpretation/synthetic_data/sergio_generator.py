"""SERGIO-based GRN data generator for interpretation training.

Uses the SERGIO simulator (Hill-kinetics ODE) to produce biologically
realistic gene expression data that closely matches DREAM4-10 benchmarks.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

# SERGIO is cloned to /tmp/SERGIO
sys.path.insert(0, "/tmp/SERGIO")
from SERGIO.sergio import sergio

from .scm_generator import SCMDataset


def _generate_scale_free_dag(
    n_genes: int,
    edge_density: float,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate a random scale-free DAG with specified density.

    Uses Barabási–Albert model, then orients edges to form a DAG.
    """
    # Target number of edges from density
    max_edges = n_genes * (n_genes - 1) // 2
    target_edges = max(n_genes - 1, int(edge_density * max_edges))
    m = max(1, round(target_edges / n_genes))
    m = min(m, n_genes - 1)

    G = nx.barabasi_albert_graph(n_genes, m, seed=int(rng.integers(0, 2**31)))

    # Orient edges: higher degree node → lower degree node (with random tiebreak)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_genes))
    ordering = list(nx.utils.arbitrary_element(nx.all_topological_sorts(
        nx.DiGraph([(u, v) for u, v in G.edges()])
    )) if False else range(n_genes))
    # Simple topological ordering by shuffled index
    order = rng.permutation(n_genes)
    rank = np.empty(n_genes, dtype=int)
    rank[order] = np.arange(n_genes)

    for u, v in G.edges():
        if rank[u] < rank[v]:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)

    # Verify it's a DAG
    assert nx.is_directed_acyclic_graph(dag), "Generated graph is not a DAG"
    return dag


def _generate_erdos_renyi_dag(
    n_genes: int,
    edge_density: float,
    rng: np.random.Generator,
) -> nx.DiGraph:
    """Generate a random Erdős–Rényi DAG."""
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_genes))
    for i in range(n_genes):
        for j in range(i + 1, n_genes):
            if rng.random() < edge_density:
                dag.add_edge(i, j)
    return dag


def generate_grn_dataset(
    n_genes: int = 10,
    n_samples: int = 200,
    edge_density: float = 0.15,
    graph_type: str = "scale_free",
    rng: np.random.Generator | None = None,
) -> SCMDataset | None:
    """Generate one GRN dataset using SERGIO.

    Returns an SCMDataset with:
    - X: expression matrix (n_samples, n_features) where n_features = n_genes - 1
    - y: expression of target gene (n_samples,)
    - dag: the regulatory network as a DAG over feature indices + target node
    - target_node: index = n_features (the last node)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate GRN topology
    if graph_type == "scale_free":
        grn = _generate_scale_free_dag(n_genes, edge_density, rng)
    else:
        grn = _generate_erdos_renyi_dag(n_genes, edge_density, rng)

    # Ensure every node has at least in-degree or out-degree > 0
    # (SERGIO requires all nodes to be either master regs or targets)
    isolated = [n for n in grn.nodes() if grn.in_degree(n) == 0 and grn.out_degree(n) == 0]
    for node in isolated:
        # Connect to a random non-isolated node
        others = [n for n in grn.nodes() if n != node and grn.out_degree(n) > 0]
        if others:
            parent = rng.choice(others)
            grn.add_edge(parent, node)

    # Identify master regulators (no incoming edges) and targets
    master_regs = [n for n in grn.nodes() if grn.in_degree(n) == 0]
    target_genes = [n for n in grn.nodes() if grn.in_degree(n) > 0]

    if len(master_regs) == 0 or len(target_genes) == 0:
        return None

    # SERGIO parameters
    # K values: positive = activation, negative = repression
    # Range based on SERGIO paper and DREAM4 literature
    k_range = (1.0, 5.0)
    coop_range = (1, 3)  # Hill coefficient
    prod_rate_range = (0.2, 2.0)  # Master regulator production rates
    decay = 0.8  # mRNA decay rate
    noise_param = 0.8  # CLE noise amplitude

    # Build interaction data for SERGIO
    # Targets file: target_idx, n_regs, reg1, ..., regN, K1, ..., KN, coop1, ..., coopN
    target_rows = []
    for t in target_genes:
        regs = list(grn.predecessors(t))
        n_regs = len(regs)
        if n_regs == 0:
            continue
        Ks = []
        coops = []
        for _ in regs:
            # ~40% repression, ~60% activation (like real GRNs)
            sign = -1 if rng.random() < 0.4 else 1
            K = sign * rng.uniform(*k_range)
            Ks.append(K)
            coops.append(rng.integers(coop_range[0], coop_range[1] + 1))

        row = [t, n_regs] + regs + Ks + [float(c) for c in coops]
        target_rows.append(row)

    # Regs file: master_reg_idx, prod_rate_bin1
    reg_rows = []
    for mr in master_regs:
        rate = rng.uniform(*prod_rate_range)
        reg_rows.append([mr, rate])

    # Write temp CSV files for SERGIO
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_targets:
        for row in target_rows:
            f_targets.write(','.join(str(x) for x in row) + '\n')
        targets_path = f_targets.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f_regs:
        for row in reg_rows:
            f_regs.write(','.join(str(x) for x in row) + '\n')
        regs_path = f_regs.name

    try:
        # Suppress SERGIO's print statements
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            sim = sergio(
                number_genes=n_genes,
                number_bins=1,  # Single cell type (steady-state, like DREAM4)
                number_sc=n_samples,
                noise_params=noise_param,
                noise_type='dpd',
                decays=decay,
                sampling_state=15,
                dt=0.01,
            )

            sim.build_graph(
                input_file_taregts=targets_path,
                input_file_regs=regs_path,
                shared_coop_state=2,  # Override coop state to 2 (standard Hill)
            )

            # Redirect stdout to suppress SERGIO's prints
            import io
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                sim.simulate()
            finally:
                sys.stdout = old_stdout

            # Get expression matrix: shape (1, n_genes, n_samples) → (n_samples, n_genes)
            expr = sim.getExpressions()
            expr = expr[0].T  # (n_samples, n_genes)

            # Clean up negative values (CLE can produce small negatives)
            expr = np.maximum(expr, 0)

    finally:
        Path(targets_path).unlink(missing_ok=True)
        Path(regs_path).unlink(missing_ok=True)

    # Sanity checks
    if np.any(np.isnan(expr)) or np.any(np.isinf(expr)):
        return None
    if expr.std() < 1e-6:
        return None

    # Pick target gene: prefer a gene with regulators (non-master-reg)
    # to ensure we have meaningful labels
    candidates = [t for t in target_genes if grn.in_degree(t) >= 1]
    if not candidates:
        return None
    target_idx = rng.choice(candidates)

    # Build feature matrix: all genes except target
    gene_indices = [g for g in range(n_genes) if g != target_idx]
    n_features = len(gene_indices)

    X = expr[:, gene_indices].astype(np.float32)
    y = expr[:, target_idx].astype(np.float32)

    # Build DAG in the feature-index space (for compute_all_labels compatibility)
    # Feature indices 0..n_features-1 map to gene_indices
    # Target node = n_features
    feature_dag = nx.DiGraph()
    feature_dag.add_nodes_from(range(n_features + 1))

    gene_to_feat = {g: i for i, g in enumerate(gene_indices)}
    gene_to_feat[target_idx] = n_features  # target node

    for u, v in grn.edges():
        if u in gene_to_feat and v in gene_to_feat:
            feature_dag.add_edge(gene_to_feat[u], gene_to_feat[v])

    # Normalize expression to [0, 1] range per gene (like DREAM4)
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-8] = 1.0
    X = (X - X_min) / X_range

    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    if y_range < 1e-8:
        return None
    y = (y - y_min) / y_range

    # Verify we have positive labels
    n_parents = sum(1 for p in feature_dag.predecessors(n_features) if p < n_features)
    if n_parents == 0:
        return None

    metadata = {
        "n_features": n_features,
        "n_samples": X.shape[0],
        "generator": "sergio",
        "graph_type": graph_type,
        "n_genes": n_genes,
        "n_edges": grn.number_of_edges(),
        "n_target_parents": n_parents,
        "target_gene_idx": int(target_idx),
        "edge_density": grn.number_of_edges() / max(1, n_genes * (n_genes - 1)),
    }

    return SCMDataset(
        X=X,
        y=y,
        dag=feature_dag,
        target_node=n_features,
        feature_names=[f"Gene_{gene_indices[i]}" for i in range(n_features)],
        edge_functions={},  # Not used for binary_direct labels
        noise_std=0.0,
        metadata=metadata,
    )
