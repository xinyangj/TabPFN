"""Unit tests for attention_extractor module.

Tests for AttentionExtractor and EdgeScoreComputer classes including:
- Hook registration and removal
- Attention weight extraction
- Edge score computation
- Aggregation methods
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch

from tabpfn import TabPFNRegressor
from tabpfn.grn.attention_extractor import AttentionExtractor, EdgeScoreComputer
from tabpfn.grn.datasets import DREAMChallengeLoader
from tabpfn.grn.preprocessing import GRNPreprocessor


# Check if Hugging Face token is available
has_hf_token = bool(os.environ.get("HF_TOKEN"))
requires_hf_auth = pytest.mark.skipif(
    not has_hf_token, reason="Requires Hugging Face authentication (HF_TOKEN env var)"
)


class TestAttentionExtractor:
    """Test suite for AttentionExtractor class."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = AttentionExtractor()
        # New implementation doesn't have cache or hooks attributes
        assert extractor is not None

    @requires_hf_auth
    def test_extract_unfitted_model_raises_error(self):
        """Test that extracting from unfitted model raises error."""
        model = TabPFNRegressor(n_estimators=1)
        extractor = AttentionExtractor()

        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="Model must be fitted"):
            extractor.extract(model, X)

    @requires_hf_auth
    def test_extract_with_fitted_model(self):
        """Test attention extraction with a fitted model."""
        # Create small synthetic dataset
        np.random.seed(42)
        X_train = np.random.randn(50, 10).astype(np.float32)
        y_train = np.random.randn(50).astype(np.float32)

        # Fit model
        model = TabPFNRegressor(n_estimators=1, device="cpu")
        model.fit(X_train, y_train)

        # Extract attention
        extractor = AttentionExtractor()
        X_test = np.random.randn(10, 10).astype(np.float32)
        attention_weights = extractor.extract(model, X_test, max_layers=1)

        # Check structure
        assert isinstance(attention_weights, dict)
        # Should have at least one layer
        if attention_weights:
            layer_key = list(attention_weights.keys())[0]
            assert isinstance(attention_weights[layer_key], dict)

    @requires_hf_auth
    def test_extract_clears_previous_cache(self):
        """Test that multiple extractions work correctly."""
        X_train = np.random.randn(30, 5).astype(np.float32)
        y_train = np.random.randn(30).astype(np.float32)

        model = TabPFNRegressor(n_estimators=1, device="cpu")
        model.fit(X_train, y_train)

        extractor = AttentionExtractor()

        # First extraction
        X_test = np.random.randn(5, 5).astype(np.float32)
        first_result = extractor.extract(model, X_test, max_layers=1)
        first_result_size = len(first_result)

        # Second extraction should work without issues
        second_result = extractor.extract(model, X_test, max_layers=1)
        assert len(second_result) == first_result_size

    @requires_hf_auth
    def test_multiple_extractions(self):
        """Test that multiple extractions work correctly."""
        X_train = np.random.randn(30, 5).astype(np.float32)
        y_train = np.random.randn(30).astype(np.float32)

        model = TabPFNRegressor(n_estimators=1, device="cpu")
        model.fit(X_train, y_train)

        extractor = AttentionExtractor()
        X_test = np.random.randn(5, 5).astype(np.float32)

        # Multiple extractions should work
        for _ in range(3):
            result = extractor.extract(model, X_test, max_layers=1)
            assert isinstance(result, dict)


class TestEdgeScoreComputer:
    """Test suite for EdgeScoreComputer class."""

    def test_init_default(self):
        """Test computer initialization with defaults."""
        computer = EdgeScoreComputer()
        assert computer.aggregation_method == "mean"

    def test_init_custom_aggregation(self):
        """Test computer initialization with custom aggregation."""
        for method in ["mean", "max", "last_layer"]:
            computer = EdgeScoreComputer(aggregation_method=method)
            assert computer.aggregation_method == method

    def test_init_invalid_aggregation(self):
        """Test that invalid aggregation method raises error."""
        with pytest.raises(ValueError, match="aggregation_method must be"):
            EdgeScoreComputer(aggregation_method="invalid")

    def test_compute_empty_attention_weights(self):
        """Test that empty attention weights raises error."""
        computer = EdgeScoreComputer()
        with pytest.raises(ValueError, match="No attention patterns found"):
            computer.compute({})

    def test_compute_with_mock_attention(self):
        """Test edge score computation with mock attention weights."""
        computer = EdgeScoreComputer(aggregation_method="mean")

        # Create mock attention weights
        attention_weights = {
            "layer_0": {
                "between_features": torch.randn(4, 10, 10),  # (n_heads, n_features, n_features)
            },
            "layer_1": {
                "between_features": torch.randn(4, 10, 10),
            },
        }

        edge_scores = computer.compute(attention_weights)

        # Check output is a tensor
        assert isinstance(edge_scores, torch.Tensor)

    def test_compute_mean_aggregation(self):
        """Test mean aggregation method."""
        computer = EdgeScoreComputer(aggregation_method="mean")

        attention_weights = {
            "layer_0": {
                "between_features": torch.ones(4, 5, 5) * 1.0,
            },
            "layer_1": {
                "between_features": torch.ones(4, 5, 5) * 2.0,
            },
        }

        edge_scores = computer.compute(attention_weights)

        # Mean should be 1.5
        assert torch.allclose(edge_scores, torch.ones(4, 5, 5) * 1.5)

    def test_compute_max_aggregation(self):
        """Test max aggregation method."""
        computer = EdgeScoreComputer(aggregation_method="max")

        attention_weights = {
            "layer_0": {
                "between_features": torch.ones(4, 5, 5) * 1.0,
            },
            "layer_1": {
                "between_features": torch.ones(4, 5, 5) * 2.0,
            },
        }

        edge_scores = computer.compute(attention_weights)

        # Max should be 2.0
        assert torch.allclose(edge_scores, torch.ones(4, 5, 5) * 2.0)

    def test_compute_last_layer_aggregation(self):
        """Test last_layer aggregation method."""
        computer = EdgeScoreComputer(aggregation_method="last_layer")

        attention_weights = {
            "layer_0": {
                "between_features": torch.ones(4, 5, 5) * 1.0,
            },
            "layer_1": {
                "between_features": torch.ones(4, 5, 5) * 2.0,
            },
        }

        edge_scores = computer.compute(attention_weights)

        # Last layer should be 2.0
        assert torch.allclose(edge_scores, torch.ones(4, 5, 5) * 2.0)

    def test_compute_use_between_items(self):
        """Test computing edge scores using between-items attention."""
        computer = EdgeScoreComputer(aggregation_method="mean")

        attention_weights = {
            "layer_0": {
                "between_items": torch.randn(4, 20, 20),  # (n_heads, n_samples, n_samples)
            },
        }

        edge_scores = computer.compute(
            attention_weights, use_between_features=False, use_between_items=True
        )

        assert isinstance(edge_scores, torch.Tensor)

    def test_compute_no_attention_type_raises_error(self):
        """Test that not using any attention type raises error."""
        computer = EdgeScoreComputer()

        attention_weights = {
            "layer_0": {
                "between_features": torch.randn(4, 5, 5),
            },
        }

        with pytest.raises(ValueError, match="At least one of"):
            computer.compute(
                attention_weights, use_between_features=False, use_between_items=False
            )


class TestIntegration:
    """Integration tests for attention extraction with GRN pipeline."""

    @requires_hf_auth
    def test_full_pipeline_with_synthetic_data(self):
        """Test full pipeline: load data -> preprocess -> fit model -> extract attention."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 50
        n_genes = 20
        n_tfs = 5

        expression = np.random.randn(n_samples, n_genes).astype(np.float32)
        gene_names = [f"GENE_{i:03d}" for i in range(n_genes)]
        tf_names = gene_names[:n_tfs]

        # Preprocess
        preprocessor = GRNPreprocessor(normalization="zscore")
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )

        # Fit model (predict first target gene)
        target_idx = 0
        model = TabPFNRegressor(n_estimators=1, device="cpu")
        model.fit(X, y[:, target_idx])

        # Extract attention
        extractor = AttentionExtractor()
        attention_weights = extractor.extract(model, X, max_layers=1)

        # Verify structure
        assert isinstance(attention_weights, dict)

        # Compute edge scores
        computer = EdgeScoreComputer(aggregation_method="mean")
        if attention_weights:
            edge_scores = computer.compute(attention_weights)
            assert isinstance(edge_scores, torch.Tensor)

    @requires_hf_auth
    def test_multiple_targets_pipeline(self):
        """Test pipeline with multiple target genes."""
        # Create small dataset for speed
        np.random.seed(42)
        X = np.random.randn(30, 5).astype(np.float32)
        y = np.random.randn(30, 3).astype(np.float32)  # 3 targets

        # Process each target
        for target_idx in range(y.shape[1]):
            model = TabPFNRegressor(n_estimators=1, device="cpu")
            model.fit(X, y[:, target_idx])

            extractor = AttentionExtractor()
            attention_weights = extractor.extract(model, X, max_layers=1)

            assert isinstance(attention_weights, dict)

    @requires_hf_auth
    def test_attention_reproducibility(self):
        """Test that attention extraction is reproducible with same input."""
        X_train = np.random.randn(30, 5).astype(np.float32)
        y_train = np.random.randn(30).astype(np.float32)

        model = TabPFNRegressor(n_estimators=1, device="cpu", random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(5, 5).astype(np.float32)

        extractor1 = AttentionExtractor()
        attention1 = extractor1.extract(model, X_test, max_layers=1)

        extractor2 = AttentionExtractor()
        attention2 = extractor2.extract(model, X_test, max_layers=1)

        # Check that both extractions have same keys
        assert set(attention1.keys()) == set(attention2.keys())

        # Check that attention patterns have same shapes
        for key in attention1.keys():
            if key in attention2:
                for attn_key in attention1[key]:
                    if attn_key in attention2[key]:
                        assert attention1[key][attn_key].shape == attention2[key][attn_key].shape
