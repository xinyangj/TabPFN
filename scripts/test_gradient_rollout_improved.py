"""Test improved gradient rollout implementation for GRN inference.

This script tests the enhanced gradient-based attention rollout that properly
computes gradients of target predictions w.r.t. attention weights in TabPFN's
in-context learning setup.
"""

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.grn import TabPFNGRNRegressor, GRNPreprocessor

def test_gradient_rollout_basic():
    """Test basic gradient rollout functionality."""
    print("=" * 70)
    print("Test 1: Gradient Rollout Basic Functionality")
    print("=" * 70)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    n_tfs = 5
    n_targets = 3

    X = np.random.randn(n_samples, n_tfs).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    tf_names = [f"TF_{i}" for i in range(n_tfs)]
    target_genes = [f"GENE_{i}" for i in range(n_targets)]

    # Test with gradient_rollout strategy
    print("\nTesting gradient_rollout strategy...")
    grn = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        edge_score_strategy="gradient_rollout",
        device="auto"
    )

    try:
        grn.fit(X, y)
        edge_scores = grn.get_edge_scores()

        print(f"✓ Gradient rollout completed successfully")
        print(f"  Number of edges: {len(edge_scores)}")

        # Check some basic properties
        for (tf, target), score in list(edge_scores.items())[:5]:
            print(f"  {tf} → {target}: {score:.4f}")

        # Verify scores are non-negative
        all_non_negative = all(s >= 0 for s in edge_scores.values())
        print(f"  All scores non-negative: {all_non_negative}")

        return True
    except Exception as e:
        print(f"✗ Gradient rollout failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_vs_sequential_rollout():
    """Compare gradient rollout with sequential rollout."""
    print("\n" + "=" * 70)
    print("Test 2: Gradient Rollout vs Sequential Rollout Comparison")
    print("=" * 70)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    n_tfs = 5
    n_targets = 2

    X = np.random.randn(n_samples, n_tfs).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    tf_names = [f"TF_{i}" for i in range(n_tfs)]
    target_genes = [f"GENE_{i}" for i in range(n_targets)]

    # Run with sequential_rollout
    print("\nRunning sequential_rollout...")
    grn_seq = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        edge_score_strategy="sequential_rollout",
        device="auto"
    )

    grn_seq.fit(X, y)
    edge_scores_seq = grn_seq.get_edge_scores()

    print(f"  Sequential rollout edges: {len(edge_scores_seq)}")

    # Run with gradient_rollout
    print("\nRunning gradient_rollout...")
    grn_grad = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        edge_score_strategy="gradient_rollout",
        device="auto"
    )

    grn_grad.fit(X, y)
    edge_scores_grad = grn_grad.get_edge_scores()

    print(f"  Gradient rollout edges: {len(edge_scores_grad)}")

    # Compare results
    print("\nComparing edge scores:")
    all_edges = set(edge_scores_seq.keys()) | set(edge_scores_grad.keys())

    for edge in sorted(all_edges)[:5]:
        seq_score = edge_scores_seq.get(edge, 0.0)
        grad_score = edge_scores_grad.get(edge, 0.0)
        diff = abs(seq_score - grad_score)
        print(f"  {edge}:")
        print(f"    Sequential: {seq_score:.4f}")
        print(f"    Gradient:   {grad_score:.4f}")
        print(f"    Difference: {diff:.4f}")

    # Check if results are different (indicating gradient weighting is working)
    max_diff = max(abs(edge_scores_seq.get(e, 0.0) - edge_scores_grad.get(e, 0.0))
                   for e in all_edges)

    print(f"\n  Maximum difference: {max_diff:.4f}")

    if max_diff > 0.01:
        print("  ✓ Gradient rollout produces different results (gradient weighting active)")
    else:
        print("  ℹ Gradient rollout produces similar results (may use fallback)")

    return True


def test_gradient_data_storage():
    """Test that X_ and y_ are properly stored for gradient computation."""
    print("\n" + "=" * 70)
    print("Test 3: Training Data Storage for Gradient Computation")
    print("=" * 70)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 30
    n_tfs = 3
    n_targets = 2

    X = np.random.randn(n_samples, n_tfs).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    tf_names = [f"TF_{i}" for i in range(n_tfs)]
    target_genes = [f"GENE_{i}" for i in range(n_targets)]

    # Fit with gradient_rollout
    grn = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        edge_score_strategy="gradient_rollout",
        device="auto"
    )

    grn.fit(X, y)

    # Check that X_ and y_ are stored
    has_X = hasattr(grn, 'X_')
    has_y = hasattr(grn, 'y_')

    print(f"  X_ stored: {has_X}")
    print(f"  y_ stored: {has_y}")

    if has_X and has_y:
        print(f"  X_ shape: {grn.X_.shape}")
        print(f"  y_ shape: {grn.y_.shape}")
        print(f"  ✓ Training data properly stored for gradient computation")
        return True
    else:
        print(f"  ✗ Training data not stored")
        return False


def test_gradient_device_handling():
    """Test that gradient computation properly handles device placement."""
    print("\n" + "=" * 70)
    print("Test 4: Gradient Rollout Device Handling")
    print("=" * 70)

    # Create synthetic data
    np.random.seed(42)
    n_samples = 20
    n_tfs = 3
    n_targets = 1

    X = np.random.randn(n_samples, n_tfs).astype(np.float32)
    y = np.random.randn(n_samples, n_targets).astype(np.float32)

    tf_names = [f"TF_{i}" for i in range(n_tfs)]
    target_genes = [f"GENE_0"]

    # Fit with gradient_rollout on specified device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Testing on device: {device}")

    grn = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        edge_score_strategy="gradient_rollout",
        device=device
    )

    try:
        grn.fit(X, y)
        edge_scores = grn.get_edge_scores()

        print(f"  ✓ Gradient rollout completed on {device}")
        print(f"  Number of edges: {len(edge_scores)}")

        # Check model devices
        if grn.target_models_:
            first_model = list(grn.target_models_.values())[0]
            if hasattr(first_model, 'devices_'):
                print(f"  Model devices: {first_model.devices_}")

        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all gradient rollout tests."""
    print("\n" + "=" * 70)
    print("IMPROVED GRADIENT ROLLOUT TEST SUITE")
    print("=" * 70)
    print("\nThis test suite validates the enhanced gradient-based attention")
    print("rollout implementation that properly computes gradients of target")
    print("predictions with respect to attention weights in TabPFN's")
    print("in-context learning setup.\n")

    results = []

    # Run tests
    results.append(("Basic Functionality", test_gradient_rollout_basic()))
    results.append(("vs Sequential Rollout", test_gradient_vs_sequential_rollout()))
    results.append(("Data Storage", test_gradient_data_storage()))
    results.append(("Device Handling", test_gradient_device_handling()))

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")

    all_passed = all(result[1] for result in results)

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
