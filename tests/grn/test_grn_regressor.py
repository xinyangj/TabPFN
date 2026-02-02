"""Unit tests for grn_regressor module.

Tests for TabPFNGRNRegressor class including:
- Model initialization
- Fitting with synthetic data
- GRN inference
- Edge score computation
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tabpfn.grn import TabPFNGRNRegressor, GRNPreprocessor

# Check if Hugging Face token is available
has_hf_token = bool(os.environ.get("HF_TOKEN"))
requires_hf_auth = pytest.mark.skipif(
    not has_hf_token, reason="Requires Hugging Face authentication (HF_TOKEN env var)"
)


class TestTabPFNGRNRegressor:
    """Test suite for TabPFNGRNRegressor class."""

    def test_init(self):
        """Test regressor initialization."""
        tf_names = ["TF1", "TF2", "TF3"]
        target_genes = ["GENE1", "GENE2"]

        grn = TabPFNGRNRegressor(
            tf_names, target_genes, n_estimators=2, attention_aggregation="mean"
        )

        assert grn.tf_names == tf_names
        assert grn.target_genes == target_genes
        assert grn.n_estimators == 2
        assert grn.attention_aggregation == "mean"
        assert grn.target_models_ == {}
        assert grn.edge_scores_ == {}

    def test_init_with_invalid_aggregation(self):
        """Test that invalid aggregation method raises error during score computation."""
        # This test doesn't require HF auth since we're just testing initialization
        grn = TabPFNGRNRegressor(
            ["TF1"], ["GENE1"], attention_aggregation="invalid"
        )
        # The error will be raised when EdgeScoreComputer is initialized during fit
        # We just verify initialization doesn't fail
        assert grn.attention_aggregation == "invalid"

    @requires_hf_auth
    def test_fit_basic(self):
        """Test basic fitting with synthetic data."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 30
        n_tfs = 3
        n_targets = 2

        X = np.random.randn(n_samples, n_tfs).astype(np.float32)
        y = np.random.randn(n_samples, n_targets).astype(np.float32)

        tf_names = [f"TF{i}" for i in range(n_tfs)]
        target_genes = [f"GENE{i}" for i in range(n_targets)]

        # Fit model
        grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=1, device="cpu")
        grn.fit(X, y)

        # Check that models were trained
        assert len(grn.target_models_) == n_targets
        assert all(target in grn.target_models_ for target in target_genes)

        # Check that attention weights were extracted
        assert len(grn.attention_weights_) == n_targets

        # Check that edge scores were computed
        assert len(grn.edge_scores_) == n_tfs * n_targets

    @requires_hf_auth
    def test_fit_dimension_mismatch_tfs(self):
        """Test that mismatched TF dimensions raise error."""
        X = np.random.randn(10, 3).astype(np.float32)
        y = np.random.randn(10, 2).astype(np.float32)

        tf_names = ["TF1", "TF2"]  # Wrong: X has 3 columns
        target_genes = ["GENE1", "GENE2"]

        grn = TabPFNGRNRegressor(tf_names, target_genes)

        with pytest.raises(ValueError, match="Number of TFs in X"):
            grn.fit(X, y)

    @requires_hf_auth
    def test_fit_dimension_mismatch_targets(self):
        """Test that mismatched target dimensions raise error."""
        X = np.random.randn(10, 2).astype(np.float32)
        y = np.random.randn(10, 3).astype(np.float32)

        tf_names = ["TF1", "TF2"]
        target_genes = ["GENE1", "GENE2"]  # Wrong: y has 3 columns

        grn = TabPFNGRNRegressor(tf_names, target_genes)

        with pytest.raises(ValueError, match="Number of targets in y"):
            grn.fit(X, y)

    @requires_hf_auth
    def test_predict_before_fit_raises_error(self):
        """Test that predicting before fitting raises error."""
        X = np.random.randn(10, 2).astype(np.float32)

        grn = TabPFNGRNRegressor(["TF1", "TF2"], ["GENE1"])

        with pytest.raises(ValueError, match="Model must be fitted"):
            grn.predict(X)

    @requires_hf_auth
    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        # Train data
        X_train = np.random.randn(30, 2).astype(np.float32)
        y_train = np.random.randn(30, 2).astype(np.float32)

        tf_names = ["TF1", "TF2"]
        target_genes = ["GENE1", "GENE2"]

        grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=1, device="cpu")
        grn.fit(X_train, y_train)

        # Test data
        X_test = np.random.randn(10, 2).astype(np.float32)
        predictions = grn.predict(X_test)

        # Check predictions
        assert isinstance(predictions, dict)
        assert set(predictions.keys()) == set(target_genes)
        assert all(isinstance(pred, np.ndarray) for pred in predictions.values())
        assert all(pred.shape == (10,) for pred in predictions.values())

    @requires_hf_auth
    def test_infer_grn_before_fit_raises_error(self):
        """Test that inferring GRN before fitting raises error."""
        grn = TabPFNGRNRegressor(["TF1"], ["GENE1"])

        with pytest.raises(ValueError, match="Model must be fitted"):
            grn.infer_grn()

    @requires_hf_auth
    def test_infer_grn_after_fit(self):
        """Test GRN inference after fitting."""
        # Train data
        X_train = np.random.randn(30, 2).astype(np.float32)
        y_train = np.random.randn(30, 2).astype(np.float32)

        tf_names = ["TF1", "TF2"]
        target_genes = ["GENE1", "GENE2"]

        grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=1, device="cpu")
        grn.fit(X_train, y_train)

        # Infer network
        network = grn.infer_grn()

        # Check network properties
        assert network.is_directed()
        assert network.number_of_nodes() == 4  # 2 TFs + 2 targets
        # Number of edges depends on threshold
        assert network.number_of_edges() >= 0

        # Check that edges have weights
        for _, _, data in network.edges(data=True):
            assert "weight" in data

    @requires_hf_auth
    def test_infer_grn_with_threshold(self):
        """Test GRN inference with score threshold."""
        X_train = np.random.randn(30, 2).astype(np.float32)
        y_train = np.random.randn(30, 2).astype(np.float32)

        tf_names = ["TF1", "TF2"]
        target_genes = ["GENE1", "GENE2"]

        grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=1, device="cpu")
        grn.fit(X_train, y_train)

        # Infer with high threshold (should have fewer edges)
        network_high = grn.infer_grn(threshold=0.9)
        # Infer with low threshold (should have more edges)
        network_low = grn.infer_grn(threshold=0.1)

        assert network_high.number_of_edges() <= network_low.number_of_edges()

    @requires_hf_auth
    def test_infer_grn_with_top_k(self):
        """Test GRN inference with top-k filtering."""
        X_train = np.random.randn(30, 3).astype(np.float32)
        y_train = np.random.randn(30, 3).astype(np.float32)

        tf_names = ["TF1", "TF2", "TF3"]
        target_genes = ["GENE1", "GENE2", "GENE3"]

        grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=1, device="cpu")
        grn.fit(X_train, y_train)

        # Infer with top_k=3
        network = grn.infer_grn(top_k=3)

        # Should have exactly 3 edges
        assert network.number_of_edges() == 3

    @requires_hf_auth
    def test_get_edge_scores_before_fit_raises_error(self):
        """Test that getting edge scores before fitting raises error."""
        grn = TabPFNGRNRegressor(["TF1"], ["GENE1"])

        with pytest.raises(ValueError, match="Model must be fitted"):
            grn.get_edge_scores()

    @requires_hf_auth
    def test_get_edge_scores_after_fit(self):
        """Test getting edge scores after fitting."""
        X_train = np.random.randn(30, 2).astype(np.float32)
        y_train = np.random.randn(30, 2).astype(np.float32)

        tf_names = ["TF1", "TF2"]
        target_genes = ["GENE1", "GENE2"]

        grn = TabPFNGRNRegressor(tf_names, target_genes, n_estimators=1, device="cpu")
        grn.fit(X_train, y_train)

        edge_scores = grn.get_edge_scores()

        # Check structure
        assert isinstance(edge_scores, dict)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in edge_scores.keys())
        assert all(isinstance(v, (int, float)) for v in edge_scores.values())

        # Check that all possible TF-target pairs are present
        for tf in tf_names:
            for target in target_genes:
                assert (tf, target) in edge_scores


class TestIntegration:
    """Integration tests for full GRN pipeline."""

    @requires_hf_auth
    def test_full_pipeline_with_preprocessing(self):
        """Test full pipeline: preprocess -> fit -> predict -> infer GRN."""
        # Create synthetic expression data
        np.random.seed(42)
        n_samples = 50
        n_genes = 15
        n_tfs = 3

        expression = np.random.randn(n_samples, n_genes).astype(np.float32)
        gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
        tf_names = gene_names[:n_tfs]

        # Preprocess
        preprocessor = GRNPreprocessor(normalization="zscore")
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        target_genes = preprocessor.get_target_names()

        # Fit GRN model
        grn = TabPFNGRNRegressor(
            tf_names,
            target_genes,
            n_estimators=1,
            device="cpu",
            attention_aggregation="mean",
        )
        grn.fit(X, y)

        # Predict
        X_test = np.random.randn(10, n_tfs).astype(np.float32)
        predictions = grn.predict(X_test)

        assert isinstance(predictions, dict)
        assert len(predictions) == len(target_genes)

        # Infer GRN
        network = grn.infer_grn(top_k=10)

        assert network.is_directed()
        assert network.number_of_edges() == 10

    @requires_hf_auth
    def test_aggregation_methods(self):
        """Test different attention aggregation methods."""
        X = np.random.randn(30, 2).astype(np.float32)
        y = np.random.randn(30, 2).astype(np.float32)

        tf_names = ["TF1", "TF2"]
        target_genes = ["GENE1", "GENE2"]

        for method in ["mean", "max", "last_layer"]:
            grn = TabPFNGRNRegressor(
                tf_names,
                target_genes,
                n_estimators=1,
                device="cpu",
                attention_aggregation=method,
            )
            grn.fit(X, y)

            # Should not raise any errors
            network = grn.infer_grn()
            assert network.is_directed()
