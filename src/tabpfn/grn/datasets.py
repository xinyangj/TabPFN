"""Dataset loaders for Gene Regulatory Network (GRN) inference.

This module provides loaders for DREAM challenge datasets used for GRN inference
evaluation and benchmarking.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import networkx as nx


class DREAMChallengeLoader:
    """Load DREAM Challenge datasets for GRN inference.

    Supports:
    - DREAM3: Smaller networks (10, 50, 100 genes)
    - DREAM4: In silico networks (10, 50, 100 genes)
    - DREAM5: Real biological networks (E. coli, S. cerevisiae, S. aureus)

    The DREAM (Dialogue for Reverse Engineering Assessments and Methods)
    challenges are the gold standard for GRN inference evaluation.

    Parameters
    ----------
    data_path : str or Path
        Path to the directory containing DREAM dataset files.

    Examples
    --------
    >>> loader = DREAMChallengeLoader(data_path='data/dream5')
    >>> expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()
    >>> print(f"Expression shape: {expression.shape}")
    """

    def __init__(self, data_path: str | Path = "data/dream5") -> None:
        """Initialize the DREAM dataset loader.

        Parameters
        ----------
        data_path : str or Path
            Path to the directory containing DREAM dataset files.
        """
        self.data_path = Path(data_path)

    def load_dream5_ecoli(
        self,
    ) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
        """Load E. coli dataset from DREAM5 challenge.

        The E. coli dataset contains:
        - 4,511 genes
        - 334 transcription factors (TFs)
        - 806 samples
        - Gold standard regulatory network

        Returns
        -------
        expression : np.ndarray
            Gene expression matrix of shape (n_samples, n_genes).
            Shape: (806, 4511)

        gene_names : list[str]
            List of gene names corresponding to columns of expression matrix.

        tf_names : list[str]
            List of transcription factor names (subset of gene_names).

        gold_standard : pd.DataFrame
            Gold standard regulatory network with columns:
            - 'tf': source transcription factor
            - 'target': target gene
            - 'weight': interaction strength (optional)

        Notes
        -----
        Expected files in data_path:
        - ecoli_expression.npy or ecoli_expression.csv
        - ecoli_genes.csv or ecoli_genes.txt
        - ecoli_tfs.csv or ecoli_tfs.txt
        - ecoli_gold_standard.csv

        If files are not found, this method will download placeholder data
        for testing purposes.
        """
        # Try to load expression data
        expression_path = self.data_path / "ecoli_expression.npy"
        if expression_path.exists():
            expression = np.load(expression_path)
        else:
            expression_path_csv = self.data_path / "ecoli_expression.csv"
            if expression_path_csv.exists():
                expression = pd.read_csv(expression_path_csv, index_col=0).values
            else:
                # Create synthetic data for testing
                warnings.warn(
                    f"E. coli expression data not found at {expression_path} or {expression_path_csv}. "
                    "Using SYNTHETIC DATA for testing. Download real DREAM5 data from: "
                    "https://dream-challenges.org/dream5/ or "
                    "https://github.com/aurineige/DREAM5/blob/master/data/DREAM5/",
                    RuntimeWarning,
                    stacklevel=2
                )
                expression = self._create_synthetic_expression(n_samples=100, n_genes=100)

        # Load gene names
        genes_path = self.data_path / "ecoli_genes.csv"
        if genes_path.exists():
            gene_names = pd.read_csv(genes_path)["gene"].tolist()
        else:
            genes_path_txt = self.data_path / "ecoli_genes.txt"
            if genes_path_txt.exists():
                gene_names = genes_path_txt.read_text().strip().split("\n")
            else:
                # Create synthetic gene names
                warnings.warn(
                    f"E. coli gene names not found at {genes_path} or {genes_path_txt}. "
                    "Using SYNTHETIC gene names.",
                    RuntimeWarning,
                    stacklevel=2
                )
                gene_names = [f"GENE_{i:04d}" for i in range(expression.shape[1])]

        # Load TF names
        tfs_path = self.data_path / "ecoli_tfs.csv"
        if tfs_path.exists():
            tf_names = pd.read_csv(tfs_path)["tf"].tolist()
        else:
            tfs_path_txt = self.data_path / "ecoli_tfs.txt"
            if tfs_path_txt.exists():
                tf_names = tfs_path_txt.read_text().strip().split("\n")
            else:
                # Use first 10% of genes as TFs for synthetic data
                warnings.warn(
                    f"E. coli TF names not found at {tfs_path} or {tfs_path_txt}. "
                    "Using SYNTHETIC TF names (first 10% of genes).",
                    RuntimeWarning,
                    stacklevel=2
                )
                n_tfs = max(10, len(gene_names) // 10)
                tf_names = gene_names[:n_tfs]

        # Load gold standard network
        gold_path = self.data_path / "ecoli_gold_standard.csv"
        if gold_path.exists():
            gold_standard = pd.read_csv(gold_path)
        else:
            # Create synthetic gold standard
            warnings.warn(
                f"E. coli gold standard not found at {gold_path}. "
                "Using SYNTHETIC gold standard network. "
                "Download real DREAM5 data from: "
                "https://dream-challenges.org/dream5/ or "
                "https://github.com/aurineige/DREAM5/blob/master/data/DREAM5/",
                RuntimeWarning,
                stacklevel=2
            )
            gold_standard = self._create_synthetic_gold_standard(tf_names, gene_names)

        return expression, gene_names, tf_names, gold_standard

    def load_dream4(
        self, network_size: int = 10, network_id: int = 1
    ) -> tuple[np.ndarray, list[str], list[str], pd.DataFrame]:
        """Load DREAM4 in silico dataset.

        DREAM4 contains simulated gene expression data with known ground-truth
        regulatory networks. These are smaller networks suitable for initial
        testing and debugging.

        Parameters
        ----------
        network_size : int, default=10
            Size of the network. Must be one of: 10, 50, 100

        network_id : int, default=1
            Network number (1-5). There are 5 networks per size.

        Returns
        -------
        expression : np.ndarray
            Gene expression matrix of shape (n_samples, n_genes).

        gene_names : list[str]
            List of gene names.

        tf_names : list[str]
            List of transcription factor names.

        gold_standard : pd.DataFrame
            Gold standard regulatory network.

        Notes
        -----
        Expected files in data_path/dream4/:
        - dream4_{size}_net{network_id}_expression.npy
        - dream4_{size}_net{network_id}_genes.csv
        - dream4_{size}_net{network_id}_tfs.csv
        - dream4_{size}_net{network_id}_gold_standard.csv

        Examples
        --------
        >>> loader = DREAMChallengeLoader(data_path='data/dream4')
        >>> expr, genes, tfs, gold = loader.load_dream4(network_size=10, network_id=1)
        """
        if network_size not in [10, 50, 100]:
            raise ValueError(f"network_size must be 10, 50, or 100, got {network_size}")
        if network_id not in range(1, 6):
            raise ValueError(f"network_id must be 1-5, got {network_id}")

        dream4_path = self.data_path / "dream4"
        prefix = f"dream4_{network_size}_net{network_id}"

        # Try to load expression data
        expression_path = dream4_path / f"{prefix}_expression.npy"
        if expression_path.exists():
            expression = np.load(expression_path)
        else:
            # Create synthetic data for testing
            warnings.warn(
                f"DREAM4 expression data not found at {expression_path}. "
                f"Using SYNTHETIC DATA for DREAM4 {network_size}-gene network (ID={network_id}). "
                "Download real DREAM4 data from: "
                "https://dream-challenges.org/dream4/ or "
                "https://github.com/aurineige/DREAM4/blob/master/data/DREAM4/",
                RuntimeWarning,
                stacklevel=2
            )
            expression = self._create_synthetic_expression(
                n_samples=100, n_genes=network_size
            )

        # Load gene names
        genes_path = dream4_path / f"{prefix}_genes.csv"
        if genes_path.exists():
            gene_names = pd.read_csv(genes_path)["gene"].tolist()
        else:
            gene_names = [f"G_{i:03d}" for i in range(network_size)]

        # Load TF names
        tfs_path = dream4_path / f"{prefix}_tfs.csv"
        if tfs_path.exists():
            tf_names = pd.read_csv(tfs_path)["tf"].tolist()
        else:
            n_tfs = max(3, network_size // 3)
            tf_names = gene_names[:n_tfs]

        # Load gold standard
        gold_path = dream4_path / f"{prefix}_gold_standard.csv"
        if gold_path.exists():
            gold_standard = pd.read_csv(gold_path)
        else:
            warnings.warn(
                f"DREAM4 gold standard not found at {gold_path}. "
                f"Using SYNTHETIC gold standard for DREAM4 {network_size}-gene network (ID={network_id}). "
                "Download real DREAM4 data from: "
                "https://dream-challenges.org/dream4/ or "
                "https://github.com/aurineige/DREAM5/blob/master/data/DREAM4/",
                RuntimeWarning,
                stacklevel=2
            )
            gold_standard = self._create_synthetic_gold_standard(tf_names, gene_names)

        return expression, gene_names, tf_names, gold_standard

    def _create_synthetic_expression(
        self, n_samples: int, n_genes: int
    ) -> np.ndarray:
        """Create synthetic gene expression data for testing.

        Parameters
        ----------
        n_samples : int
            Number of samples

        n_genes : int
            Number of genes

        Returns
        -------
        expression : np.ndarray
            Synthetic expression matrix of shape (n_samples, n_genes)
        """
        np.random.seed(42)
        # Create expression with some correlation structure
        base_signal = np.random.randn(n_samples, 1)
        noise = np.random.randn(n_samples, n_genes) * 0.5
        expression = base_signal + noise
        return expression.astype(np.float32)

    def _create_synthetic_gold_standard(
        self, tf_names: list[str], gene_names: list[str]
    ) -> pd.DataFrame:
        """Create synthetic gold standard network for testing.

        Creates random TF-target edges with approximately 10% sparsity.

        Parameters
        ----------
        tf_names : list[str]
            List of transcription factor names

        gene_names : list[str]
            List of all gene names

        Returns
        -------
        gold_standard : pd.DataFrame
            DataFrame with columns: 'tf', 'target', 'weight'
        """
        tf_set = set(tf_names)
        target_genes = [g for g in gene_names if g not in tf_set]

        # Create random edges (about 10% of possible)
        np.random.seed(42)
        edges = []
        n_edges = int(len(tf_names) * len(target_genes) * 0.1)

        for _ in range(n_edges):
            tf = np.random.choice(tf_names)
            target = np.random.choice(target_genes)
            weight = np.random.uniform(0.5, 1.0)
            edges.append({"tf": tf, "target": target, "weight": weight})

        return pd.DataFrame(edges)

    def get_gold_standard_network(
        self, gold_standard: pd.DataFrame
    ) -> "nx.DiGraph":
        """Convert gold standard DataFrame to NetworkX graph.

        Parameters
        ----------
        gold_standard : pd.DataFrame
            Gold standard network with 'tf', 'target', 'weight' columns

        Returns
        -------
        network : nx.DiGraph
            Directed graph representation of the gold standard
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for graph operations. "
                "Install with: pip install networkx"
            )

        graph = nx.DiGraph()
        for _, row in gold_standard.iterrows():
            graph.add_edge(
                row["tf"],
                row["target"],
                weight=row.get("weight", 1.0),
            )
        return graph
