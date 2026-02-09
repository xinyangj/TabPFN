#!/usr/bin/env python3
"""Simple test for vectorized attention rollout implementation.

This script verifies that the vectorized functions produce identical results
to the original loop-based implementation using smaller test sizes.
"""

import torch
import numpy as np
import time


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


def test_build_features_attention_matrix():
    """Test that vectorized version produces identical results."""
    print("Testing build_features_attention_matrix...")

    num_items = 5
    num_feature_blocks = 3
    feat_attn = torch.randn(num_items, num_items)

    # Original version
    start = time.time()
    A_orig = build_features_attention_matrix_original(feat_attn, num_items, num_feature_blocks)
    orig_time = time.time() - start

    # Vectorized version
    start = time.time()
    J = torch.ones(num_feature_blocks, num_feature_blocks, device=feat_attn.device, dtype=feat_attn.dtype)
    A_vec = torch.kron(feat_attn, J)
    vec_time = time.time() - start

    # Check equality
    if torch.allclose(A_orig, A_vec, rtol=1e-5, atol=1e-7):
        print(f"  ✓ PASSED - Speedup: {orig_time/vec_time:.1f}x")
        return True
    else:
        print(f"  ✗ FAILED - Max diff: {(A_orig - A_vec).abs().max().item()}")
        return False


def test_build_items_attention_matrix():
    """Test that vectorized version produces identical results."""
    print("Testing build_items_attention_matrix...")

    num_items = 4
    num_feature_blocks = 3
    item_attn = torch.randn(num_feature_blocks, num_feature_blocks)

    # Original version
    start = time.time()
    A_orig = build_items_attention_matrix_original(item_attn, num_items, num_feature_blocks)
    orig_time = time.time() - start

    # Vectorized version
    start = time.time()
    J = torch.ones(num_items, num_items, device=item_attn.device, dtype=item_attn.dtype)
    A_vec = torch.kron(J, item_attn)
    vec_time = time.time() - start

    # Check equality
    if torch.allclose(A_orig, A_vec, rtol=1e-5, atol=1e-7):
        print(f"  ✓ PASSED - Speedup: {orig_time/vec_time:.1f}x")
        return True
    else:
        print(f"  ✗ FAILED - Max diff: {(A_orig - A_vec).abs().max().item()}")
        return False


def main():
    print("=" * 60)
    print("TESTING VECTORIZED ATTENTION ROLLOUT")
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

    # Summary
    print("\n" + "=" * 60)
    print(f"TESTS PASSED: {tests_passed}/{tests_total}")
    print("=" * 60)

    if tests_passed == tests_total:
        print("✓ All tests passed! Vectorized implementation is correct.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    exit(main())
