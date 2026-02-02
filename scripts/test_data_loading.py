#!/usr/bin/env python
"""Test data loading and preprocessing pipeline for GRN inference.

This script tests the full data loading pipeline including:
1. Loading DREAM challenge datasets
2. Preprocessing expression data
3. Preparing TF and target matrices
4. Verifying data formats

Usage:
    python scripts/test_data_loading.py
"""

import sys
from pathlib import Path

import numpy as np

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from tabpfn.grn.datasets import DREAMChallengeLoader
from tabpfn.grn.preprocessing import GRNPreprocessor


def test_dream4_pipeline():
    """Test the full pipeline with DREAM4 data (small networks for testing)."""
    print("=" * 60)
    print("Testing DREAM4 Pipeline")
    print("=" * 60)

    # Initialize loader
    loader = DREAMChallengeLoader(data_path="data/dream4")

    # Test different network sizes
    for size in [10, 50]:
        for network_id in [1, 2]:
            print(f"\n--- DREAM4: size={size}, network_id={network_id} ---")

            # Load data
            expression, gene_names, tf_names, gold_standard = loader.load_dream4(
                network_size=size, network_id=network_id
            )

            print(f"Expression shape: {expression.shape}")
            print(f"Number of genes: {len(gene_names)}")
            print(f"Number of TFs: {len(tf_names)}")
            print(f"Gold standard edges: {len(gold_standard)}")

            # Preprocess
            preprocessor = GRNPreprocessor(normalization="zscore")
            X, y, tf_indices, target_indices = preprocessor.fit_transform(
                expression, gene_names, tf_names
            )

            print(f"X shape (TF features): {X.shape}")
            print(f"y shape (target genes): {y.shape}")
            print(f"Number of TF indices: {len(tf_indices)}")
            print(f"Number of target indices: {len(target_indices)}")

            # Verify data
            assert X.shape[0] == y.shape[0], "Sample count mismatch"
            assert len(tf_indices) == len(tf_names), "TF count mismatch"
            assert len(target_indices) == len(gene_names) - len(tf_names), "Target count mismatch"

            # Check normalization
            print(f"X mean: {X.mean():.4f}, std: {X.std():.4f}")
            print(f"y mean: {y.mean():.4f}, std: {y.std():.4f}")

    print("\n✓ DREAM4 pipeline test passed!")


def test_dream5_pipeline():
    """Test the full pipeline with DREAM5 data."""
    print("\n" + "=" * 60)
    print("Testing DREAM5 Pipeline")
    print("=" * 60)

    # Initialize loader
    loader = DREAMChallengeLoader(data_path="data/dream5")

    # Load E. coli data
    print("\n--- DREAM5: E. coli ---")
    expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

    print(f"Expression shape: {expression.shape}")
    print(f"Number of genes: {len(gene_names)}")
    print(f"Number of TFs: {len(tf_names)}")
    print(f"Gold standard edges: {len(gold_standard)}")

    # Preprocess
    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )

    print(f"X shape (TF features): {X.shape}")
    print(f"y shape (target genes): {y.shape}")
    print(f"Number of TF indices: {len(tf_indices)}")
    print(f"Number of target indices: {len(target_indices)}")

    # Verify data
    assert X.shape[0] == y.shape[0], "Sample count mismatch"
    assert len(tf_indices) == len(tf_names), "TF count mismatch"

    # Check normalization
    print(f"X mean: {X.mean():.4f}, std: {X.std():.4f}")
    print(f"y mean: {y.mean():.4f}, std: {y.std():.4f}")

    # Test transform with new data
    print("\n--- Testing transform with new data ---")
    expression_new = np.random.randn(50, expression.shape[1]).astype(np.float32)
    X_new, y_new = preprocessor.transform(expression_new, gene_names)

    print(f"New X shape: {X_new.shape}")
    print(f"New y shape: {y_new.shape}")
    assert X_new.shape[1] == X.shape[1], "Feature count mismatch"

    print("\n✓ DREAM5 pipeline test passed!")


def test_normalization_methods():
    """Test different normalization methods."""
    print("\n" + "=" * 60)
    print("Testing Normalization Methods")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    expression = np.random.randn(100, 50).astype(np.float32)
    gene_names = [f"GENE_{i:03d}" for i in range(50)]
    tf_names = gene_names[:10]

    methods = ["none", "zscore", "log", "quantile"]

    for method in methods:
        print(f"\n--- Testing {method} normalization ---")

        # Make data positive for log normalization
        if method == "log":
            expression_test = expression - expression.min() + 1.0
        else:
            expression_test = expression

        preprocessor = GRNPreprocessor(normalization=method)
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression_test, gene_names, tf_names
        )

        print(f"X shape: {X.shape}, mean: {X.mean():.4f}, std: {X.std():.4f}")
        print(f"y shape: {y.shape}, mean: {y.mean():.4f}, std: {y.std():.4f}")

        # Verify no NaN or Inf
        assert np.all(np.isfinite(X)), f"NaN or Inf found in X with {method}"
        assert np.all(np.isfinite(y)), f"NaN or Inf found in y with {method}"

    print("\n✓ Normalization methods test passed!")


def test_interaction_features():
    """Test interaction features."""
    print("\n" + "=" * 60)
    print("Testing Interaction Features")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    expression = np.random.randn(100, 50).astype(np.float32)
    gene_names = [f"GENE_{i:03d}" for i in range(50)]
    tf_names = gene_names[:10]

    # Without interactions
    print("\n--- Without interaction features ---")
    preprocessor_no_interact = GRNPreprocessor(
        normalization="none", add_interaction_features=False
    )
    X_no_interact, y, tf_indices, target_indices = (
        preprocessor_no_interact.fit_transform(expression, gene_names, tf_names)
    )
    print(f"X shape: {X_no_interact.shape}")

    # With interactions
    print("\n--- With interaction features ---")
    preprocessor_with_interact = GRNPreprocessor(
        normalization="none", add_interaction_features=True
    )
    X_with_interact, _, _, _ = preprocessor_with_interact.fit_transform(
        expression, gene_names, tf_names
    )
    print(f"X shape: {X_with_interact.shape}")

    # Calculate expected number of interactions
    n_tfs = len(tf_names)
    n_interactions = n_tfs * (n_tfs - 1) // 2
    expected_n_features = n_tfs + n_interactions

    assert (
        X_with_interact.shape[1] == expected_n_features
    ), f"Expected {expected_n_features} features, got {X_with_interact.shape[1]}"
    assert X_no_interact.shape[1] == n_tfs

    print(f"\nBase features: {n_tfs}")
    print(f"Interaction features: {n_interactions}")
    print(f"Total features: {expected_n_features}")

    print("\n✓ Interaction features test passed!")


def test_gold_standard_network():
    """Test gold standard network conversion."""
    print("\n" + "=" * 60)
    print("Testing Gold Standard Network")
    print("=" * 60)

    loader = DREAMChallengeLoader(data_path="data/dream4")
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=10, network_id=1
    )

    # Convert to NetworkX graph
    graph = loader.get_gold_standard_network(gold_standard)

    print(f"Graph nodes: {graph.number_of_nodes()}")
    print(f"Graph edges: {graph.number_of_edges()}")
    print(f"Is directed: {graph.is_directed()}")

    # Verify graph properties
    assert graph.is_directed()
    assert graph.number_of_edges() == len(gold_standard)

    # Check some edges
    print(f"\nSample edges:")
    for i, (tf, target, data) in enumerate(graph.edges(data=True)):
        if i < 5:
            print(f"  {tf} -> {target} (weight: {data.get('weight', 1.0):.3f})")

    print("\n✓ Gold standard network test passed!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TabPFN GRN - Data Loading Pipeline Test")
    print("=" * 60)

    try:
        test_dream4_pipeline()
        test_dream5_pipeline()
        test_normalization_methods()
        test_interaction_features()
        test_gold_standard_network()

        print("\n" + "=" * 60)
        print("All tests passed successfully! ✓")
        print("=" * 60 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
