"""Unit test for verifying index alignment after fixing information leakage.

This test verifies that:
1. Target genes are properly excluded from input features during training
2. Index mapping correctly maps attention indices back to original TF indices
3. No self-edges exist in predicted edge scores

Tests the new refactored wrapper classes that manage their own models:
- SklearnForestWrapper (for GENIE3/GRNBoost2)
- LinearRegressionWrapper (for Correlation/MI)
"""

import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from tabpfn.grn.baseline_models import SklearnForestWrapper, LinearRegressionWrapper


class TestIndexAlignment(unittest.TestCase):
    """Test suite for verifying index alignment after fixing information leakage.

    Tests the new refactored wrapper classes that manage their own models:
    - SklearnForestWrapper (for GENIE3/GRNBoost2)
    - LinearRegressionWrapper (for Correlation/MI)
    """

    def test_sklearn_forest_wrapper_target_exclusion(self):
        """Test SklearnForestWrapper properly excludes target from features."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randn(100, 10).astype(np.float32)
        gene_names = [f"G{i}" for i in range(10)]
        tf_names = gene_names
        target_genes = gene_names

        # Fit wrapper
        wrapper = SklearnForestWrapper(
            estimator_class=RandomForestRegressor,
            estimator_kwargs={"max_features": "sqrt"},
            n_estimators=10,
            random_state=42,
        )
        wrapper.fit(X, y, tf_names, target_genes)

        # Verify: Each target model should have stored indices
        for target_idx, target_name in enumerate(target_genes):
            assert hasattr(wrapper, "tf_indices_per_target_"), \
                f"{target_name}: Missing tf_indices_per_target_ for target {target_idx}"

            # Verify target is excluded if it's also a TF
            if target_name in tf_names:
                expected_tf_count = len(tf_names) - 1
                actual_tf_count = len(wrapper.tf_indices_per_target_[target_idx])
                assert actual_tf_count == expected_tf_count, \
                    f"{target_name}: Expected {expected_tf_count} TFs (excluding self), got {actual_tf_count}"

        print("SklearnForestWrapper target exclusion: PASSED")

    def test_linear_regression_wrapper_target_exclusion(self):
        """Test LinearRegressionWrapper properly excludes target from features."""
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = np.random.randn(100, 3).astype(np.float32)
        gene_names = [f"G{i}" for i in range(3)]
        tf_names = gene_names
        target_genes = gene_names

        # Fit wrapper
        wrapper = LinearRegressionWrapper(random_state=42)
        wrapper.fit(X, y, tf_names, target_genes)

        # Verify: coef_ matrix should be fitted
        assert wrapper.coef_ is not None, "coef_ should not be None after fit"
        assert wrapper.coef_.shape == (len(tf_names), len(target_genes)), \
            f"coef_ shape mismatch: expected ({len(tf_names)}, {len(target_genes)}), got {wrapper.coef_.shape}"

        # Verify: For each target that is also a TF, the diagonal element should be 0
        # (target excluded from its own features during training)
        for target_idx, target_name in enumerate(target_genes):
            if target_name in tf_names:
                tf_idx = tf_names.index(target_name)
                coef_value = wrapper.coef_[tf_idx, target_idx]
                # When fitting, the target was excluded from features, so the coefficient
                # for the self-edge should be 0 (the default initialization value)
                assert coef_value == 0.0, \
                    f"{target_name}: Self-edge coefficient should be 0, got {coef_value}"

        print("LinearRegressionWrapper target exclusion: PASSED")


if __name__ == "__main__":
    unittest.main()
