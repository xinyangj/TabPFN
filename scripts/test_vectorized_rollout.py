#!/usr/bin/env python3
"""Test vectorized attention rollout implementation.

This script verifies that the vectorized functions produce identical results
to the original loop-based implementation.
"""

import torch
import numpy as np


def build_features_attention_matrix_original(
    feat_attn: torch.Tensor,
    num_items: int,
    num_feature_blocks: int,
) -> torch.Tensor:
    """Original loop-based implementation."""
    N = num_items * num_feature_blocks
    A_feat = torch.zeros(N, N, device=feat_attn.device, dtype=feat_attn.dtype)

    for i in range(num_items):
        for j in range(num_items):
            row_start = i * num_feature_blocks
            row_end = (i + 1) * num_feature_blocks
            col_start = j * num_feature_blocks
            col_end = (j + 1) * num_feature_blocks
            A_feat[row_start:row_end, col_start:col_end] = feat_attn[i, j]

    return A_feat


def build_items_attention_matrix_original(
    item_attn: torch.Tensor,
    num_items: int,
    num_feature_blocks: int,
) -> torch.Tensor:
    """Original loop-based implementation."""
    N = num_items * num_feature_blocks
    A_items = torch.zeros(N, N, device=item_attn.device, dtype=item_attn.dtype)

    for fi in range(num_feature_blocks):
        for fj in range(num_feature_blocks):
            for i in range(num_items):
                for j in range(num_items):
                    row_pos = i * num_feature_blocks + fi
                    col_pos = j * num_feature_blocks + fj
                    A_items[row_pos, col_pos] = item_attn[fi, fj]

    return A_items


def extract_edge_scores_from_rollout_original(
    rollout_matrix: torch.Tensor,
    tf_indices: list[int],
    target_idx: int,
    num_items: int,
    num_feature_blocks: int,
) -> dict[int, float]:
    """Original loop-based implementation."""
    edge_scores = {}

    for tf_idx in tf_indices:
        tf_target_scores = []

        for item_idx in range(num_items):
            tf_pos = item_idx * num_feature_blocks + tf_idx
            target_pos = item_idx * num_feature_blocks + target_idx

            if tf_pos < rollout_matrix.size(0) and target_pos < rollout_matrix.size(1):
                tf_target_scores.append(rollout_matrix[tf_pos, target_pos].item())

        edge_scores[tf_idx] = np.mean(tf_target_scores) if tf_target_scores else 0.0

    return edge_scores


def test_build_features_attention_matrix():
    """Test that vectorized version produces identical results."""
    print("Testing build_features_attention_matrix...")

    num_items = 10
    num_feature_blocks = 5
    feat_attn = torch.randn(num_items, num_items)

    # Original version
    A_orig = build_features_attention_matrix_original(feat_attn, num_items, num_feature_blocks)

    # Vectorized version
    J = torch.ones(num_feature_blocks, num_feature_blocks, device=feat_attn.device, dtype=feat_attn.dtype)
    A_vec = torch.kron(feat_attn, J)

    # Check equality
    if torch.allclose(A_orig, A_vec, rtol=1e-5, atol=1e-7):
        print("  ✓ PASSED: Vectorized version produces identical results")
        return True
    else:
        print("  ✗ FAILED: Results differ")
        print(f"    Max diff: {(A_orig - A_vec).abs().max().item()}")
        return False


def test_build_items_attention_matrix():
    """Test that vectorized version produces identical results."""
    print("Testing build_items_attention_matrix...")

    num_items = 8
    num_feature_blocks = 6
    item_attn = torch.randn(num_feature_blocks, num_feature_blocks)

    # Original version
    A_orig = build_items_attention_matrix_original(item_attn, num_items, num_feature_blocks)

    # Vectorized version
    J = torch.ones(num_items, num_items, device=item_attn.device, dtype=item_attn.dtype)
    A_vec = torch.kron(J, item_attn)

    # Check equality
    if torch.allclose(A_orig, A_vec, rtol=1e-5, atol=1e-7):
        print("  ✓ PASSED: Vectorized version produces identical results")
        return True
    else:
        print("  ✗ FAILED: Results differ")
        print(f"    Max diff: {(A_orig - A_vec).abs().max().item()}")
        return False


def test_extract_edge_scores_from_rollout():
    """Test that vectorized version produces identical results."""
    print("Testing extract_edge_scores_from_rollout...")

    num_items = 20
    num_feature_blocks = 10
    N = num_items * num_feature_blocks
    rollout_matrix = torch.randn(N, N)
    tf_indices = [0, 2, 5, 7]
    target_idx = 9

    # Original version
    scores_orig = extract_edge_scores_from_rollout_original(
        rollout_matrix, tf_indices, target_idx, num_items, num_feature_blocks
    )

    # Vectorized version
    device = rollout_matrix.device
    tf_indices_tensor = torch.tensor(tf_indices, device=device, dtype=torch.long)
    item_indices = torch.arange(num_items, device=device)
    tf_positions = item_indices[:, None] * num_feature_blocks + tf_indices_tensor[None, :]
    target_positions = item_indices * num_feature_blocks + target_idx
    all_scores = rollout_matrix[tf_positions, target_positions[:, None]]
    mean_scores = all_scores.mean(dim=0)
    scores_vec = {tf_idx: mean_scores[i].item() for i, tf_idx in enumerate(tf_indices)}

    # Check equality
    all_match = True
    for tf_idx in tf_indices:
        if abs(scores_orig[tf_idx] - scores_vec[tf_idx]) > 1e-6:
            print(f"  ✗ FAILED: TF {tf_idx} differs: {scores_orig[tf_idx]} vs {scores_vec[tf_idx]}")
            all_match = False

    if all_match:
        print("  ✓ PASSED: Vectorized version produces identical results")
        return True
    else:
        return False


def benchmark_functions():
    """Benchmark the speedup of vectorized functions."""
    print("\n" + "=" * 60)
    print("BENCHMARKING SPEEDUP")
    print("=" * 60)

    import time

    # Test various sizes
    sizes = [
        (10, 5, "Small"),
        (50, 20, "Medium"),
        (100, 50, "Large"),
    ]

    for num_items, num_feature_blocks, label in sizes:
        print(f"\n{label}: num_items={num_items}, num_feature_blocks={num_feature_blocks}")

        # Benchmark build_features_attention_matrix
        feat_attn = torch.randn(num_items, num_items)

        start = time.time()
        for _ in range(100):
            A_orig = build_features_attention_matrix_original(feat_attn, num_items, num_feature_blocks)
        orig_time = time.time() - start

        start = time.time()
        for _ in range(100):
            J = torch.ones(num_feature_blocks, num_feature_blocks, device=feat_attn.device, dtype=feat_attn.dtype)
            A_vec = torch.kron(feat_attn, J)
        vec_time = time.time() - start

        speedup = orig_time / vec_time if vec_time > 0 else float('inf')
        print(f"  build_features: {orig_time:.4f}s → {vec_time:.4f}s ({speedup:.1f}x speedup)")

        # Benchmark build_items_attention_matrix
        item_attn = torch.randn(num_feature_blocks, num_feature_blocks)

        start = time.time()
        for _ in range(10):
            A_orig = build_items_attention_matrix_original(item_attn, num_items, num_feature_blocks)
        orig_time = time.time() - start

        start = time.time()
        for _ in range(100):
            J = torch.ones(num_items, num_items, device=item_attn.device, dtype=item_attn.dtype)
            A_vec = torch.kron(J, item_attn)
        vec_time = time.time() - start

        speedup = orig_time / vec_time if vec_time > 0 else float('inf')
        print(f"  build_items:   {orig_time:.4f}s → {vec_time:.4f}s ({speedup:.1f}x speedup)")


def main():
    print("=" * 60)
    print("TESTING VECTORIZED ATTENTION ROLLOUT IMPLEMENTATION")
    print("=" * 60)

    # Run tests
    tests_passed = 0
    tests_total = 0

    # Test 1: build_features_attention_matrix
    tests_total += 1
    if test_build_features_attention_matrix():
        tests_passed += 1

    # Test 2: build_items_attention_matrix
    tests_total += 1
    if test_build_items_attention_matrix():
        tests_passed += 1

    # Test 3: extract_edge_scores_from_rollout
    tests_total += 1
    if test_extract_edge_scores_from_rollout():
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"TESTS PASSED: {tests_passed}/{tests_total}")
    print("=" * 60)

    if tests_passed == tests_total:
        print("✓ All tests passed! Vectorized implementation is correct.")
    else:
        print("✗ Some tests failed. Please check the implementation.")

    # Benchmark
    benchmark_functions()


if __name__ == "__main__":
    main()
