#!/usr/bin/env python
"""Analyze the actual shapes of attention tensors returned by EdgeScoreComputer.

This script helps understand the structure of attention weights so we can
properly extract TF-target edge scores.
"""

import numpy as np
import torch
from tabpfn.grn import DREAMChallengeLoader, GRNPreprocessor, TabPFNGRNRegressor
from tabpfn.grn.attention_extractor import AttentionExtractor, EdgeScoreComputer


def main():
    print("="*70)
    print("Analyzing Attention Tensor Shapes")
    print("="*70)

    # Load a small dataset
    loader = DREAMChallengeLoader()
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=10, network_id=1
    )

    print(f"\nDataset info:")
    print(f"  Expression shape: {expression.shape}")
    print(f"  Gene names: {len(gene_names)}")
    print(f"  TF names: {len(tf_names)}")
    print(f"  Gold standard edges: {len(gold_standard)}")

    # Preprocess
    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )
    target_genes = preprocessor.get_target_names()

    print(f"\nAfter preprocessing:")
    print(f"  X shape (TFs): {X.shape}")
    print(f"  y shape (targets): {y.shape}")
    print(f"  TF names: {len(tf_names)}")
    print(f"  Target genes: {len(target_genes)}")

    # Train a simple model for one target
    print(f"\n" + "="*70)
    print("Training TabPFN for ONE target gene")
    print("="*70)

    target_idx = 0
    target_name = target_genes[target_idx]

    print(f"\nTarget: {target_name}")
    print(f"Input X shape: {X.shape}")
    print(f"Target y shape: {y[:, target_idx].shape}")

    # Create and fit a TabPFNRegressor directly
    from tabpfn import TabPFNRegressor

    model = TabPFNRegressor(n_estimators=1, device="cpu")
    model.fit(X, y[:, target_idx])

    # Extract attention
    print(f"\n" + "="*70)
    print("Extracting Attention Weights")
    print("="*70)

    extractor = AttentionExtractor()
    attention = extractor.extract(model, X, max_layers=2)

    print(f"\nAttention keys: {list(attention.keys())}")

    for layer_key, layer_data in attention.items():
        print(f"\n{layer_key}:")
        for attn_key, attn_tensor in layer_data.items():
            print(f"  {attn_key}: shape={attn_tensor.shape}, dtype={attn_tensor.dtype}")
            # Print some stats
            print(f"    min={attn_tensor.min().item():.6f}, "
                  f"max={attn_tensor.max().item():.6f}, "
                  f"mean={attn_tensor.mean().item():.6f}")

    # Use EdgeScoreComputer
    print(f"\n" + "="*70)
    print("Testing EdgeScoreComputer")
    print("="*70)

    computer = EdgeScoreComputer(aggregation_method="mean")
    aggregated = computer.compute(
        attention,
        use_between_features=True,
        use_between_items=False,
    )

    print(f"\nAggregated attention shape: {aggregated.shape}")
    print(f"Aggregated attention dtype: {aggregated.dtype}")
    print(f"min={aggregated.min().item():.6f}, "
          f"max={aggregated.max().item():.6f}, "
          f"mean={aggregated.mean().item():.6f}")

    # Try to understand the relationship between TFs and attention
    print(f"\n" + "="*70)
    print("Understanding TF-Attention Relationship")
    print("="*70)

    print(f"\nNumber of TFs: {len(tf_names)}")
    print(f"Number of targets: 1 (just {target_name})")
    print(f"X shape: {X.shape} = (n_samples={X.shape[0]}, n_TFs={X.shape[1]})")

    # The key question: How do we map attention to TF-target scores?
    print(f"\nPossible interpretations:")
    print(f"  1. If attention is (n_TFs, n_TFs), use diagonal or row/column means")
    print(f"  2. If attention has sample dimension, aggregate across samples")
    print(f"  3. If attention has feature groups, need to map to TFs")

    # Check if we can find a correlation
    print(f"\nAttempting to extract TF-specific scores:")

    # Try different interpretations
    if aggregated.ndim == 2:
        print(f"  2D tensor: {aggregated.shape}")
        if aggregated.shape[0] == len(tf_names):
            print(f"    First dim matches n_TFs! Using row means.")
            scores = aggregated.mean(dim=1).detach().cpu().numpy()
        elif aggregated.shape[1] == len(tf_names):
            print(f"    Second dim matches n_TFs! Using column means.")
            scores = aggregated.mean(dim=0).detach().cpu().numpy()
        else:
            print(f"    Neither dimension matches n_TFs. Using flattened mean.")
            scores = aggregated.mean().detach().cpu().numpy()
            scores = np.full(len(tf_names), scores)
    elif aggregated.ndim == 3:
        print(f"  3D tensor: {aggregated.shape}")
        # Try aggregating across first dimension
        scores = aggregated.mean(dim=(0, 1)).detach().cpu().numpy()
        if len(scores) != len(tf_names):
            scores = aggregated.mean().detach().cpu().numpy()
            scores = np.full(len(tf_names), scores)
    elif aggregated.ndim == 4:
        print(f"  4D tensor: {aggregated.shape}")
        scores = aggregated.mean(dim=(0, 1, 2)).detach().cpu().numpy()
        if len(scores) != len(tf_names):
            scores = aggregated.mean().detach().cpu().numpy()
            scores = np.full(len(tf_names), scores)
    else:
        print(f"  {aggregated.ndim}D tensor: using single value for all TFs")
        scores = np.full(len(tf_names), aggregated.mean().item())

    print(f"\nExtracted scores (first 10 TFs):")
    for i, (tf, score) in enumerate(zip(tf_names[:10], scores[:10])):
        print(f"  {tf}: {score:.6f}")

    # Check if scores vary (non-random)
    print(f"\nScore statistics:")
    print(f"  min: {scores.min():.6f}")
    print(f"  max: {scores.max():.6f}")
    print(f"  mean: {scores.mean():.6f}")
    print(f"  std: {scores.std():.6f}")
    print(f"  Are all same? {np.allclose(scores, scores[0])}")

    print(f"\n" + "="*70)
    print("Analysis Complete")
    print("="*70)
    print(f"\nKey findings:")
    print(f"  - Attention tensor shape: {aggregated.shape}")
    print(f"  - Number of TFs: {len(tf_names)}")
    print(f"  - Score variance: {scores.std():.6f}")
    print(f"\nNext steps:")
    print(f"  1. Implement proper TF-target score extraction based on shape analysis")
    print(f"  2. Replace np.random.random() placeholder in grn_regressor.py")
    print(f"  3. Test with actual GRN evaluation")


if __name__ == "__main__":
    main()
