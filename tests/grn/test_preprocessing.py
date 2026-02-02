"""Unit tests for preprocessing module.

Tests for GRNPreprocessor class including:
- TF and target identification
- Expression normalization
- Feature engineering
- Transform operations
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from tabpfn.grn.preprocessing import GRNPreprocessor


class TestGRNPreprocessor:
    """Test suite for GRNPreprocessor class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample gene expression data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_genes = 50
        n_tfs = 10

        expression = np.random.randn(n_samples, n_genes).astype(np.float32)
        gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
        tf_names = gene_names[:n_tfs]

        return expression, gene_names, tf_names

    def test_init(self):
        """Test preprocessor initialization."""
        preprocessor = GRNPreprocessor()
        assert preprocessor.normalization == "zscore"
        assert preprocessor.add_interaction_features is False

        preprocessor_log = GRNPreprocessor(normalization="log")
        assert preprocessor_log.normalization == "log"

        preprocessor_interact = GRNPreprocessor(add_interaction_features=True)
        assert preprocessor_interact.add_interaction_features is True

    def test_init_invalid_normalization(self):
        """Test that invalid normalization raises error."""
        with pytest.raises(ValueError, match="normalization must be"):
            GRNPreprocessor(normalization="invalid")

    def test_fit_transform_basic(self, sample_data):
        """Test basic fit_transform operation."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor(normalization="none")

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Check X (TF features)
        assert X.shape[0] == expression.shape[0]  # n_samples
        assert X.shape[1] == len(tf_names)  # n_TFs

        # Check y (targets)
        assert y.shape[0] == expression.shape[0]  # n_samples
        assert y.shape[1] == len(gene_names) - len(tf_names)  # n_targets

        # Check indices
        assert len(tf_indices) == len(tf_names)
        assert len(target_indices) == len(gene_names) - len(tf_names)

    def test_fit_transform_zscore_normalization(self, sample_data):
        """Test z-score normalization."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor(normalization="zscore")

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Check that normalized data has mean ~0 and std ~1
        np.testing.assert_array_almost_equal(X.mean(axis=0), 0.0, decimal=5)
        np.testing.assert_array_almost_equal(X.std(axis=0), 1.0, decimal=5)

        # Same for y
        np.testing.assert_array_almost_equal(y.mean(axis=0), 0.0, decimal=5)
        np.testing.assert_array_almost_equal(y.std(axis=0), 1.0, decimal=5)

    def test_fit_transform_no_normalization(self, sample_data):
        """Test without normalization."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor(normalization="none")

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Check that data is unchanged
        np.testing.assert_array_equal(
            X[:, : len(tf_names)], expression[:, tf_indices]
        )

    def test_fit_transform_log_normalization(self, sample_data):
        """Test log normalization."""
        expression, gene_names, tf_names = sample_data
        # Make all values positive for log
        expression_pos = expression - expression.min() + 1.0

        preprocessor = GRNPreprocessor(normalization="log")

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression_pos, gene_names, tf_names
        )

        # Check that values are transformed (all positive)
        assert np.all(X >= 0)
        assert np.all(y >= 0)

    def test_fit_transform_with_interaction_features(self, sample_data):
        """Test with interaction features enabled."""
        expression, gene_names, tf_names = sample_data
        n_tfs = len(tf_names)
        n_interactions = n_tfs * (n_tfs - 1) // 2

        preprocessor = GRNPreprocessor(
            normalization="none", add_interaction_features=True
        )

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Check that interaction features are added
        expected_n_features = n_tfs + n_interactions
        assert X.shape[1] == expected_n_features

    def test_fit_transform_invalid_input_shape(self):
        """Test that invalid input shape raises error."""
        preprocessor = GRNPreprocessor()

        # 3D input
        expression_3d = np.random.randn(10, 5, 3)
        gene_names = [f"GENE_{i}" for i in range(5)]
        tf_names = gene_names[:2]

        with pytest.raises(ValueError, match="must be 2D"):
            preprocessor.fit_transform(expression_3d, gene_names, tf_names)

    def test_fit_transform_mismatched_dimensions(self):
        """Test that mismatched dimensions raise error."""
        preprocessor = GRNPreprocessor()

        expression = np.random.randn(10, 5)
        gene_names = [f"GENE_{i}" for i in range(10)]  # Wrong!
        tf_names = gene_names[:2]

        with pytest.raises(ValueError, match="Number of gene names.*must match"):
            preprocessor.fit_transform(expression, gene_names, tf_names)

    def test_fit_transform_no_tfs(self):
        """Test that having no TFs raises error."""
        preprocessor = GRNPreprocessor()

        expression = np.random.randn(10, 5)
        gene_names = [f"GENE_{i}" for i in range(5)]
        tf_names = []  # No TFs

        with pytest.raises(ValueError, match="No TFs found"):
            preprocessor.fit_transform(expression, gene_names, tf_names)

    def test_fit_transform_no_targets(self):
        """Test that having no targets raises error."""
        preprocessor = GRNPreprocessor()

        expression = np.random.randn(10, 5)
        gene_names = [f"GENE_{i}" for i in range(5)]
        tf_names = gene_names[:]  # All genes are TFs

        with pytest.raises(ValueError, match="No target genes found"):
            preprocessor.fit_transform(expression, gene_names, tf_names)

    def test_get_tf_names_before_fit(self):
        """Test that get_tf_names returns None before fitting."""
        preprocessor = GRNPreprocessor()
        assert preprocessor.get_tf_names() is None

    def test_get_tf_names_after_fit(self, sample_data):
        """Test that get_tf_names returns TF names after fitting."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor()

        preprocessor.fit_transform(expression, gene_names, tf_names)

        assert preprocessor.get_tf_names() == tf_names

    def test_get_target_names_after_fit(self, sample_data):
        """Test that get_target_names returns target names after fitting."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor()

        preprocessor.fit_transform(expression, gene_names, tf_names)

        expected_targets = [g for g in gene_names if g not in tf_names]
        assert preprocessor.get_target_names() == expected_targets

    def test_transform_before_fit_raises_error(self):
        """Test that transform before fit raises error."""
        preprocessor = GRNPreprocessor()

        expression = np.random.randn(10, 5)
        gene_names = [f"GENE_{i}" for i in range(5)]

        with pytest.raises(ValueError, match="must be fitted before transform"):
            preprocessor.transform(expression, gene_names)

    def test_transform_after_fit(self, sample_data):
        """Test transform with new data using fitted parameters."""
        expression_train, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor(normalization="zscore")

        # Fit on training data
        preprocessor.fit_transform(expression_train, gene_names, tf_names)

        # Transform new data
        expression_test = np.random.randn(50, 50).astype(np.float32)
        X_test, y_test = preprocessor.transform(expression_test, gene_names)

        # Check shapes
        assert X_test.shape[0] == 50  # n_samples_test
        assert X_test.shape[1] == len(tf_names)  # n_TFs
        assert y_test.shape[0] == 50
        assert y_test.shape[1] == len(gene_names) - len(tf_names)

    def test_interaction_features_formula(self, sample_data):
        """Test that interaction features are computed correctly."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor(
            normalization="none", add_interaction_features=True
        )

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Manually compute first interaction feature
        n_tfs = len(tf_names)
        tf_expression = expression[:, tf_indices]

        # First interaction should be TF[0] * TF[1]
        expected_first_interaction = tf_expression[:, 0] * tf_expression[:, 1]

        # Find the first interaction in X (after the n_tfs base features)
        actual_first_interaction = X[:, n_tfs]

        np.testing.assert_array_almost_equal(
            actual_first_interaction, expected_first_interaction
        )

    def test_zscore_zero_std_handling(self):
        """Test that zero std is handled correctly in zscore normalization."""
        # Create data with one constant feature
        expression = np.random.randn(100, 10).astype(np.float32)
        expression[:, 0] = 1.0  # Constant feature (first TF)

        gene_names = [f"GENE_{i}" for i in range(10)]
        tf_names = gene_names[:5]

        preprocessor = GRNPreprocessor(normalization="zscore")

        # Should not raise error
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Constant feature should remain finite (or be all zeros)
        assert np.all(np.isfinite(X[:, 0]))

    def test_all_tfs_are_in_gene_names(self, sample_data):
        """Test that all TFs are subset of gene_names."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor()

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Check that stored TF names are all in gene names
        tf_names_stored = preprocessor.get_tf_names()
        assert all(tf in gene_names for tf in tf_names_stored)

    def test_reproducibility_same_seed(self, sample_data):
        """Test that results are reproducible with same input."""
        expression, gene_names, tf_names = sample_data
        preprocessor1 = GRNPreprocessor()
        preprocessor2 = GRNPreprocessor()

        X1, y1, tf_indices1, target_indices1 = preprocessor1.fit_transform(
            expression, gene_names, tf_names
        )
        X2, y2, tf_indices2, target_indices2 = preprocessor2.fit_transform(
            expression, gene_names, tf_names
        )

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
        assert tf_indices1 == tf_indices2
        assert target_indices1 == target_indices2

    def test_quantile_normalization_properties(self, sample_data):
        """Test quantile normalization properties."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor(normalization="quantile")

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Check that values are in [0, 1) range
        assert np.all(X >= 0)
        assert np.all(X < 1)
        assert np.all(y >= 0)
        assert np.all(y < 1)

    def test_target_genes_exclude_tfs(self, sample_data):
        """Test that target genes don't include TFs."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor()

        _, _, _, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        target_names = [gene_names[i] for i in target_indices]
        tf_set = set(tf_names)

        # No target should be a TF
        assert not any(target in tf_set for target in target_names)

    def test_tf_and_target_indices_are_correct(self, sample_data):
        """Test that TF and target indices correctly map to gene_names."""
        expression, gene_names, tf_names = sample_data
        preprocessor = GRNPreprocessor()

        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Verify TF indices
        for idx in tf_indices:
            assert gene_names[idx] in tf_names

        # Verify target indices
        for idx in target_indices:
            assert gene_names[idx] not in tf_names

        # Verify no overlap
        assert set(tf_indices).isdisjoint(set(target_indices))
        assert set(tf_indices) | set(target_indices) == set(range(len(gene_names)))
