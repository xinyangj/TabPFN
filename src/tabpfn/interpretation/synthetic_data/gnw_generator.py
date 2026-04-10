"""GeneNetWeaver-based data generator for interpretation training.

Uses GNW (the same tool that generated the DREAM4 benchmarks) to produce
biologically realistic gene expression data with thermodynamic TF-binding
kinetics, protein+mRNA dynamics, and perturbation experiments.

Two modes:
  1. Bio mode: Extract subnetworks from real organism regulatory networks
  2. Synthetic mode: Generate random DAGs, convert to GNW format, simulate

Both modes use GNW's full ODE/SDE simulation with microarray noise.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np

from .scm_generator import SCMDataset

# Paths to GNW resources (relative to this repo)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_GNW_DIR = _REPO_ROOT / "data" / "gnw"
_GNW_JAR = _GNW_DIR / "gnw-3.1.2b.jar"
_GNW_SETTINGS = _GNW_DIR / "settings.txt"
_GNW_SETTINGS_D4S10 = _GNW_DIR / "settings_dream4_size10.txt"

# Source networks for bio mode (default: our collection)
_SOURCE_NETWORKS = {
    "ecoli_regulondb": _GNW_DIR / "ecoli_transcriptional_network_regulonDB_6_7.tsv",
    "yeast_400": _GNW_DIR / "yeast_400_net3.tsv",
    "ecoli_1200": _GNW_DIR / "ecoli_1200_net4.tsv",
}

# Original DREAM4 source networks (RegulonDB 6.2 + Yeast Balaji 2006)
_SOURCE_NETWORKS_DREAM4 = {
    "ecoli_regulondb_6_2": _GNW_DIR / "ecoli_transcriptional_network_regulonDB_6_2.tsv",
    "yeast_balaji2006": _GNW_DIR / "yeast_transcriptional_network_Balaji2006.tsv",
}


def _run_gnw(args: list[str], cwd: str, timeout: int = 60) -> str:
    """Run GNW jar with given arguments. Returns stdout+stderr."""
    cmd = ["java", "-jar", str(_GNW_JAR)] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.stdout + result.stderr


def _generate_random_dag(
    n_genes: int,
    rng: np.random.Generator,
    graph_type: str = "erdos_renyi",
    min_edges: int | None = None,
    max_edges: int | None = None,
    min_tfs: int | None = None,
    dream4_size10: bool = False,
) -> nx.DiGraph:
    """Generate a random DAG matching DREAM4-10 scale.

    For 10-gene networks: targets ~12-16 edges, ≥7 active TFs.
    When dream4_size10=True: tightens to exact DREAM4-10 range (12-16 edges).
    """
    if dream4_size10:
        # Exact DREAM4-10 gold standard range: 12, 13, 15, 15, 16
        if min_edges is None:
            min_edges = 12
        if max_edges is None:
            max_edges = 16
        if min_tfs is None:
            min_tfs = max(2, int(0.7 * n_genes))
    else:
        if min_edges is None:
            min_edges = max(n_genes, int(0.10 * n_genes * (n_genes - 1)))
        if max_edges is None:
            max_edges = int(0.22 * n_genes * (n_genes - 1))
        if min_tfs is None:
            min_tfs = max(2, int(0.6 * n_genes))

    for _attempt in range(200):
        if graph_type == "erdos_renyi":
            # p=0.30 for directed, ~half survive DAG orientation
            p = 0.30 if n_genes <= 15 else 0.15
            G = nx.erdos_renyi_graph(
                n_genes, p, directed=True,
                seed=int(rng.integers(0, 2**31)),
            )
        else:  # scale_free
            m = max(1, min(3, n_genes // 4))
            G = nx.barabasi_albert_graph(
                n_genes, m, seed=int(rng.integers(0, 2**31)),
            )
            # Convert undirected to directed
            G = G.to_directed()

        G.remove_edges_from(nx.selfloop_edges(G))

        # Orient edges to form DAG using random topological order
        order = rng.permutation(n_genes)
        rank = np.empty(n_genes, dtype=int)
        rank[order] = np.arange(n_genes)

        dag = nx.DiGraph()
        dag.add_nodes_from(range(n_genes))
        for u, v in G.edges():
            if rank[u] < rank[v]:
                dag.add_edge(u, v)

        n_edges = dag.number_of_edges()
        active_tfs = sum(1 for nd in dag.nodes() if dag.out_degree(nd) > 0)

        if min_edges <= n_edges <= max_edges and active_tfs >= min_tfs:
            return dag

    # Fallback: return last attempt regardless
    return dag


def _dag_to_tsv(
    dag: nx.DiGraph,
    path: str,
    rng: np.random.Generator,
    activation_prob: float = 0.6,
) -> None:
    """Write a DAG as GNW-compatible TSV with random edge signs."""
    with open(path, "w") as f:
        for u, v in dag.edges():
            sign = "+" if rng.random() < activation_prob else "-"
            f.write(f"G{u+1}\tG{v+1}\t{sign}\n")


def _parse_timeseries(path: str) -> np.ndarray:
    """Parse GNW time-series TSV → (n_samples, n_genes) array, dropping time column."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            parts = line.split("\t")
            try:
                vals = [float(x) for x in parts]
                rows.append(vals[1:])  # drop time column
            except ValueError:
                continue
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 0))


def _parse_steady_state(path: str) -> np.ndarray:
    """Parse GNW steady-state TSV → (n_samples, n_genes) array."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('"'):
                continue
            parts = line.split("\t")
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return np.array(rows, dtype=np.float32) if rows else np.empty((0, 0))


def _parse_gold_standard(path: str) -> list[tuple[str, str, int]]:
    """Parse GNW gold standard TSV → list of (tf, target, weight)."""
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                edges.append((parts[0], parts[1], int(parts[2])))
    return edges


def _parse_gene_names(timeseries_path: str) -> list[str]:
    """Extract gene names from the header of a GNW TSV file."""
    with open(timeseries_path) as f:
        header = f.readline().strip()
    # Header formats:
    #   "Time"\tG1\tG2\t...  (time-series, gene names unquoted)
    #   G1\tG2\t...           (steady-state, no Time column)
    #   "Time"\t"G1"\t"G2"   (some GNW versions quote everything)
    parts = header.split("\t")
    names = []
    for p in parts:
        p = p.strip().strip('"')
        if p and p != "Time":
            names.append(p)
    return names


def generate_gnw_dataset(
    n_genes: int = 10,
    mode: Literal["bio", "synthetic"] = "synthetic",
    source_network: str | None = None,
    graph_type: str = "erdos_renyi",
    rng: np.random.Generator | None = None,
    dream4_size10: bool = False,
) -> SCMDataset | None:
    """Generate one GRN dataset using GeneNetWeaver.

    Parameters
    ----------
    n_genes : int
        Number of genes in the network.
    mode : "bio" or "synthetic"
        "bio": extract subnetwork from a real organism network.
        "synthetic": generate random DAG, simulate with GNW dynamics.
    source_network : str, optional
        For bio mode: which source network to use.
        If None, randomly chosen from available sources.
    graph_type : str
        For synthetic mode: "erdos_renyi" or "scale_free".
    rng : np.random.Generator, optional
        Random number generator.
    dream4_size10 : bool
        If True, align generation parameters with actual DREAM4 Size10:
        - Use original DREAM4 source networks (RegulonDB 6.2 + Yeast Balaji 2006)
        - 40% E. coli / 60% Yeast source distribution
        - 5 time series (not 10)
        - Edge range 12-16 for synthetic mode
        - Variable activation/repression ratio (33-53%)
        - No self-interactions in bio mode
        - Prefer cycles in subnetwork extraction

    Returns
    -------
    SCMDataset or None
        Dataset with expression matrix, labels, and DAG, or None on failure.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not _GNW_JAR.exists():
        raise FileNotFoundError(f"GNW jar not found at {_GNW_JAR}")

    # Select settings file and source networks based on alignment mode
    settings_path = _GNW_SETTINGS_D4S10 if dream4_size10 else _GNW_SETTINGS
    source_pool = _SOURCE_NETWORKS_DREAM4 if dream4_size10 else _SOURCE_NETWORKS

    with tempfile.TemporaryDirectory(prefix="gnw_") as tmpdir:
        net_name = "GNWNet"

        if mode == "bio":
            # Pick source network
            if source_network is None:
                if dream4_size10:
                    # DREAM4-10: 2/5 E. coli, 3/5 Yeast → 40%/60%
                    source_network = (
                        "ecoli_regulondb_6_2" if rng.random() < 0.4
                        else "yeast_balaji2006"
                    )
                else:
                    source_network = rng.choice(list(source_pool.keys()))
            src_path = source_pool.get(source_network)
            if src_path is None or not src_path.exists():
                return None

            # Extract subnetwork
            extract_args = [
                "--extract",
                "-c", str(settings_path),
                "--input-net", str(src_path),
                "--random-seed",
                "--greedy-selection",
                f"--subnet-size={n_genes}",
                "--num-subnets=1",
                "--output-net-format=4",
                "--output-path", tmpdir,
                "--network-name", net_name,
            ]
            if not dream4_size10:
                # Default mode: keep self-interactions
                extract_args.append("--keep-self-interactions")
            output = _run_gnw(extract_args, cwd=tmpdir, timeout=30)

            sbml_path = os.path.join(tmpdir, f"{net_name}-1.xml")
            if not os.path.exists(sbml_path):
                return None
            net_name = f"{net_name}-1"

        else:  # synthetic
            dag = _generate_random_dag(
                n_genes, rng, graph_type, dream4_size10=dream4_size10,
            )
            tsv_path = os.path.join(tmpdir, "network.tsv")
            # DREAM4-10 activation ratio: 33-53% (sampled per network)
            activation_prob = (
                rng.uniform(0.33, 0.53) if dream4_size10 else 0.6
            )
            _dag_to_tsv(dag, tsv_path, rng, activation_prob=activation_prob)

            # Convert TSV → SBML with kinetic parameters
            output = _run_gnw([
                "--transform",
                "-c", str(settings_path),
                "--input-net", tsv_path,
                "--output-net-format=4",
                "--output-path", tmpdir,
                "--network-name", net_name,
            ], cwd=tmpdir, timeout=30)

            sbml_path = os.path.join(tmpdir, f"{net_name}.xml")
            if not os.path.exists(sbml_path):
                return None

        # Simulate
        output = _run_gnw([
            "--simulate",
            "-c", str(settings_path),
            "--input-net", sbml_path,
        ], cwd=tmpdir, timeout=120)

        # Parse outputs (GNW writes to cwd = tmpdir)
        ts_path = os.path.join(tmpdir, f"{net_name}_dream4_timeseries.tsv")
        ko_path = os.path.join(tmpdir, f"{net_name}_knockouts.tsv")
        kd_path = os.path.join(tmpdir, f"{net_name}_knockdowns.tsv")
        mf_path = os.path.join(tmpdir, f"{net_name}_multifactorial.tsv")
        gs_path = os.path.join(tmpdir, f"{net_name}_goldstandard.tsv")

        if not os.path.exists(gs_path):
            return None

        # Parse gene names
        for p in [ts_path, ko_path, mf_path]:
            if os.path.exists(p):
                gene_names = _parse_gene_names(p)
                break
        else:
            return None

        actual_n_genes = len(gene_names)
        if actual_n_genes < 3:
            return None

        # Stack all experiment types as samples
        data_parts = []
        if os.path.exists(ts_path):
            ts = _parse_timeseries(ts_path)
            if ts.size > 0:
                data_parts.append(ts)
        if os.path.exists(ko_path):
            ko = _parse_steady_state(ko_path)
            if ko.size > 0:
                data_parts.append(ko)
        if os.path.exists(kd_path):
            kd = _parse_steady_state(kd_path)
            if kd.size > 0:
                data_parts.append(kd)
        if os.path.exists(mf_path):
            mf = _parse_steady_state(mf_path)
            if mf.size > 0:
                data_parts.append(mf)

        if not data_parts:
            return None

        expr = np.vstack(data_parts)  # (n_total_samples, n_genes)

        # Parse gold standard
        gold_edges = _parse_gold_standard(gs_path)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Pick target gene: prefer genes with regulators
        targets_with_regs = []
        for tf, tg, w in gold_edges:
            if w == 1 and tg in gene_to_idx:
                targets_with_regs.append(tg)
        targets_with_regs = list(set(targets_with_regs))

        if not targets_with_regs:
            return None

        target_gene = rng.choice(targets_with_regs)
        target_idx = gene_to_idx[target_gene]

        # Build feature matrix (all genes except target)
        feature_indices = [i for i in range(actual_n_genes) if i != target_idx]
        n_features = len(feature_indices)

        X = expr[:, feature_indices]
        y = expr[:, target_idx]

        # Build DAG in feature-index space
        feature_dag = nx.DiGraph()
        feature_dag.add_nodes_from(range(n_features + 1))

        feat_gene_names = [gene_names[i] for i in feature_indices]
        feat_name_to_idx = {g: i for i, g in enumerate(feat_gene_names)}

        n_parents = 0
        for tf, tg, w in gold_edges:
            if w == 1:
                if tg == target_gene and tf in feat_name_to_idx:
                    # Edge from feature to target
                    feature_dag.add_edge(feat_name_to_idx[tf], n_features)
                    n_parents += 1
                elif tf in feat_name_to_idx and tg in feat_name_to_idx:
                    # Edge between features
                    feature_dag.add_edge(
                        feat_name_to_idx[tf], feat_name_to_idx[tg]
                    )

        if n_parents == 0:
            return None

        # Sanity checks
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            return None
        if X.std() < 1e-8:
            return None

        metadata = {
            "n_features": n_features,
            "n_samples": X.shape[0],
            "generator": "gnw",
            "mode": mode,
            "source_network": source_network if mode == "bio" else graph_type,
            "n_genes": actual_n_genes,
            "n_edges": sum(1 for _, _, w in gold_edges if w == 1),
            "n_target_parents": n_parents,
            "target_gene": target_gene,
            "edge_density": sum(1 for _, _, w in gold_edges if w == 1)
            / max(1, actual_n_genes * (actual_n_genes - 1)),
            "dream4_aligned": dream4_size10,
        }

        return SCMDataset(
            X=X,
            y=y,
            dag=feature_dag,
            target_node=n_features,
            feature_names=feat_gene_names,
            edge_functions={},
            noise_std=0.0,
            metadata=metadata,
        )
