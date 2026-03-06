"""Benchmark comparisons for interpretation model evaluation.

Compares the trained interpretation model against baseline methods
for feature importance estimation.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def random_baseline(n_features: int) -> np.ndarray:
    """Random importance baseline — assigns uniform random scores."""
    return np.random.random(n_features)


def attention_heuristic_baseline(
    signals: dict[str, Any],
    n_features: int,
) -> np.ndarray:
    """Attention-to-target heuristic baseline.

    Uses mean attention from each feature to the target block across
    all layers and heads — the simplest interpretation of TabPFN attention.
    """
    attn_dict = signals.get("between_features_attention", {})
    if not attn_dict:
        return np.zeros(n_features)

    scores = np.zeros(n_features)
    n_counted = 0

    for key, attn in sorted(attn_dict.items()):
        if attn is None:
            continue
        attn_np = attn.cpu().float().numpy()
        # attn shape: (batch_items, n_heads, n_blocks, n_blocks)
        # Average over batch and heads
        attn_mean = attn_np.mean(axis=(0, 1))  # (n_blocks, n_blocks)
        n_blocks = attn_mean.shape[0]
        target_block = n_blocks - 1

        for feat_idx in range(min(n_features, target_block)):
            block_idx = min(feat_idx, target_block - 1)
            scores[feat_idx] += attn_mean[block_idx, target_block]
        n_counted += 1

    if n_counted > 0:
        scores /= n_counted

    return scores


def gradient_magnitude_baseline(
    signals: dict[str, Any],
    n_features: int,
) -> np.ndarray:
    """Input gradient magnitude baseline.

    Uses |∂prediction/∂X_i| as a feature importance proxy.
    """
    input_grads = signals.get("input_gradients")
    if input_grads is None:
        return np.zeros(n_features)

    grads = input_grads.cpu().float().numpy()
    if grads.ndim == 3:
        grads = grads[:, 0, :]  # Remove batch dim

    # Mean absolute gradient across samples
    scores = np.abs(grads).mean(axis=0)
    if len(scores) > n_features:
        scores = scores[:n_features]
    elif len(scores) < n_features:
        scores = np.pad(scores, (0, n_features - len(scores)))

    # Normalize
    max_score = scores.max()
    if max_score > 0:
        scores /= max_score

    return scores
