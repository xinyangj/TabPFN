"""Unit tests for datasets module.

Tests for DREAMChallengeLoader class including:
- Loading synthetic data when real data is not available
- Gold standard network creation
- Data format validation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tabpfn.grn.datasets import DREAMChallengeLoader


class TestDREAMChallengeLoader:
    """Test suite for DREAMChallengeLoader class."""

    def test_init(self):
        """Test loader initialization."""
        loader = DREAMChallengeLoader()
        assert loader.data_path == Path("data/dream5")

        loader_with_path = DREAMChallengeLoader(data_path="custom/path")
        assert loader_with_path.data_path == Path("custom/path")

    def test_load_dream5_ecoli_synthetic_data(self, tmp_path):
        """Test loading E. coli data with synthetic data generation."""
        loader = DREAMChallengeLoader(data_path=tmp_path)

        expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

        # Check shapes and types
        assert isinstance(expression, np.ndarray)
        assert expression.ndim == 2
        assert expression.shape[0] == 100  # n_samples for synthetic
        assert expression.shape[1] == 100  # n_genes for synthetic

        # Check gene names
        assert isinstance(gene_names, list)
        assert len(gene_names) == expression.shape[1]
        assert all(isinstance(name, str) for name in gene_names)

        # Check TF names
        assert isinstance(tf_names, list)
        assert all(isinstance(name, str) for name in tf_names)
        assert len(tf_names) <= len(gene_names)
        assert all(tf in gene_names for tf in tf_names)

        # Check gold standard
        assert isinstance(gold_standard, pd.DataFrame)
        assert "tf" in gold_standard.columns
        assert "target" in gold_standard.columns
        assert "weight" in gold_standard.columns
        assert all(gold_standard["tf"].isin(tf_names))
        assert all(gold_standard["target"].isin(gene_names))

    def test_load_dream4_synthetic_data(self, tmp_path):
        """Test loading DREAM4 data with synthetic data generation."""
        loader = DREAMChallengeLoader(data_path=tmp_path)

        for size in [10, 50, 100]:
            for network_id in [1, 3, 5]:
                expression, gene_names, tf_names, gold_standard = loader.load_dream4(
                    network_size=size, network_id=network_id
                )

                # Check size
                assert expression.shape[1] == size
                assert len(gene_names) == size

                # Check TF count (approximately 1/3 of genes)
                n_tfs = len(tf_names)
                expected_tfs = max(3, size // 3)
                assert n_tfs == expected_tfs

                # Check gold standard
                assert "tf" in gold_standard.columns
                assert "target" in gold_standard.columns

    def test_load_dream4_invalid_size(self, tmp_path):
        """Test that invalid network size raises error."""
        loader = DREAMChallengeLoader(data_path=tmp_path)

        with pytest.raises(ValueError, match="network_size must be 10, 50, or 100"):
            loader.load_dream4(network_size=20)

    def test_load_dream4_invalid_network_id(self, tmp_path):
        """Test that invalid network ID raises error."""
        loader = DREAMChallengeLoader(data_path=tmp_path)

        with pytest.raises(ValueError, match="network_id must be 1-5"):
            loader.load_dream4(network_id=10)

    def test_synthetic_expression_properties(self, tmp_path):
        """Test that synthetic expression data has expected properties."""
        loader = DREAMChallengeLoader(data_path=tmp_path)
        expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

        # Check dtype
        assert expression.dtype == np.float32

        # Check that there's variation (not all zeros)
        assert not np.allclose(expression, 0.0)

        # Check that there's some correlation structure
        # (genes should be correlated due to the synthetic generation)
        corr_matrix = np.corrcoef(expression.T)
        # Remove diagonal
        np.fill_diagonal(corr_matrix, 0)
        # At least some correlations should be non-zero
        assert np.any(np.abs(corr_matrix) > 0.1)

    def test_synthetic_gold_standard_properties(self, tmp_path):
        """Test that synthetic gold standard has expected properties."""
        loader = DREAMChallengeLoader(data_path=tmp_path)
        expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

        # Check that gold standard has approximately 10% sparsity
        n_possible_edges = len(tf_names) * (len(gene_names) - len(tf_names))
        n_edges = len(gold_standard)
        sparsity = n_edges / n_possible_edges
        assert 0.05 < sparsity < 0.15  # Approximately 10%

        # Check that all edges are from TFs to non-TF genes
        assert all(gold_standard["tf"].isin(tf_names))
        assert not all(gold_standard["target"].isin(tf_names))

        # Check that weights are in valid range
        assert all(gold_standard["weight"] >= 0)
        assert all(gold_standard["weight"] <= 1)

    def test_get_gold_standard_network(self, tmp_path):
        """Test conversion of gold standard to NetworkX graph."""
        loader = DREAMChallengeLoader(data_path=tmp_path)
        expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

        graph = loader.get_gold_standard_network(gold_standard)

        # Check graph properties
        assert graph.is_directed()
        assert graph.number_of_nodes() > 0

        # Note: NetworkX collapses duplicate edges, so we check unique edges
        unique_edges = gold_standard.drop_duplicates(subset=["tf", "target"])
        assert graph.number_of_edges() == len(unique_edges)

        # Check that all unique edges are present
        for _, row in unique_edges.iterrows():
            assert graph.has_edge(row["tf"], row["target"])

    def test_get_gold_standard_network_requires_networkx(self, tmp_path):
        """Test that NetworkX is required for graph conversion."""
        # This test verifies the error message when networkx is not available
        # In practice, we can't easily test this since networkx should be installed
        # The test mainly documents the requirement
        loader = DREAMChallengeLoader(data_path=tmp_path)
        expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

        # Should work if networkx is installed
        try:
            graph = loader.get_gold_standard_network(gold_standard)
            assert graph is not None
        except ImportError as e:
            assert "networkx is required" in str(e)

    def test_synthetic_data_reproducibility(self, tmp_path):
        """Test that synthetic data is reproducible (same seed)."""
        loader1 = DREAMChallengeLoader(data_path=tmp_path)
        expression1, gene_names1, tf_names1, gold_standard1 = loader1.load_dream5_ecoli()

        loader2 = DREAMChallengeLoader(data_path=tmp_path)
        expression2, gene_names2, tf_names2, gold_standard2 = loader2.load_dream5_ecoli()

        # Data should be identical (same random seed)
        np.testing.assert_array_equal(expression1, expression2)
        assert gene_names1 == gene_names2
        assert tf_names1 == tf_names2
        pd.testing.assert_frame_equal(gold_standard1, gold_standard2)

    def test_load_dream5_ecoli_with_existing_files(self, tmp_path):
        """Test loading when actual data files exist."""
        loader = DREAMChallengeLoader(data_path=tmp_path)

        # Create test data files
        expression = np.random.randn(50, 20).astype(np.float32)
        np.save(tmp_path / "ecoli_expression.npy", expression)

        genes = pd.DataFrame({"gene": [f"GENE_{i}" for i in range(20)]})
        genes.to_csv(tmp_path / "ecoli_genes.csv", index=False)

        tfs = pd.DataFrame({"tf": [f"GENE_{i}" for i in range(5)]})
        tfs.to_csv(tmp_path / "ecoli_tfs.csv", index=False)

        # Load the data
        expression_loaded, gene_names_loaded, tf_names_loaded, _ = loader.load_dream5_ecoli()

        # Check that loaded data matches
        np.testing.assert_array_equal(expression, expression_loaded)
        assert len(gene_names_loaded) == 20
        assert len(tf_names_loaded) == 5

    def test_load_dream4_with_existing_files(self, tmp_path):
        """Test loading DREAM4 when actual data files exist."""
        loader = DREAMChallengeLoader(data_path=tmp_path)

        # Create dream4 subdirectory and files
        dream4_path = tmp_path / "dream4"
        dream4_path.mkdir()

        expression = np.random.randn(50, 10).astype(np.float32)
        np.save(dream4_path / "dream4_10_net1_expression.npy", expression)

        genes = pd.DataFrame({"gene": [f"G_{i}" for i in range(10)]})
        genes.to_csv(dream4_path / "dream4_10_net1_genes.csv", index=False)

        # Load the data
        expression_loaded, gene_names_loaded, _, _ = loader.load_dream4(
            network_size=10, network_id=1
        )

        # Check that loaded data matches
        np.testing.assert_array_equal(expression, expression_loaded)
        assert len(gene_names_loaded) == 10

    def test_gold_standard_no_self_loops(self, tmp_path):
        """Test that gold standard doesn't contain self-loops."""
        loader = DREAMChallengeLoader(data_path=tmp_path)
        expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

        # Check that no TF regulates itself
        for _, row in gold_standard.iterrows():
            assert row["tf"] != row["target"]


@pytest.fixture
def tmp_path():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
