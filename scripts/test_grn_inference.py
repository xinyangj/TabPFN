#!/usr/bin/env python
"""Test GRN inference with actual attention weights."""

import numpy as np
from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn,
)

def main():
    print("="*70)
    print("Testing GRN Inference with Actual Attention Weights")
    print("="*70)

    # Load DREAM4 dataset
    loader = DREAMChallengeLoader()
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=10, network_id=1
    )

    print(f"\nDataset: DREAM4-10-network1")
    print(f"  Expression shape: {expression.shape}")
    print(f"  Genes: {len(gene_names)}, TFs: {len(tf_names)}")
    print(f"  Gold standard edges: {len(gold_standard)}")

    # Preprocess
    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )
    target_genes = preprocessor.get_target_names()

    print(f"\nAfter preprocessing:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Target genes: {len(target_genes)}")

    # Train GRN model
    print(f"\nTraining TabPFN GRN model...")
    grn_model = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        attention_aggregation="mean",
        device="cpu",
    )
    grn_model.fit(X, y)

    # Get edge scores
    edge_scores = grn_model.get_edge_scores()
    print(f"\nEdge scores:")
    print(f"  Number of edges: {len(edge_scores)}")

    # Analyze edge scores
    scores_array = np.array(list(edge_scores.values()))
    print(f"  Min: {scores_array.min():.6f}")
    print(f"  Max: {scores_array.max():.6f}")
    print(f"  Mean: {scores_array.mean():.6f}")
    print(f"  Std: {scores_array.std():.6f}")
    print(f"  All same? {np.allclose(scores_array, scores_array[0])}")

    # Show top edges
    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 edges:")
    for (tf, tgt), score in sorted_edges[:10]:
        is_correct = (tf, tgt) in gold_standard
        marker = "✓" if is_correct else " "
        print(f"  {marker} {tf} -> {tgt}: {score:.6f}")

    # Evaluate
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=[5, 10])
    print(f"\nEvaluation metrics:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    for k in [5, 10]:
        if f"precision@{k}" in metrics:
            print(f"  Precision@{k}:  {metrics[f'precision@{k}']:.4f}")
        if f"recall@{k}" in metrics:
            print(f"  Recall@{k}:     {metrics[f'recall@{k}']:.4f}")

    # Check if AUPR is non-random
    if metrics['aupr'] > 0:
        print(f"\n✓ SUCCESS: AUPR > 0 ({metrics['aupr']:.4f})")
        print("  Edge scores are based on actual attention weights, not random!")
    else:
        print(f"\n✗ WARNING: AUPR = 0 ({metrics['aupr']:.4f})")
        print("  Edge scores may still be random or not properly extracted.")

    # Check for variation in scores
    if scores_array.std() > 0:
        print(f"\n✓ SUCCESS: Score variation detected (std={scores_array.std():.6f})")
        print("  Different TFs have different edge scores!")
    else:
        print(f"\n✗ WARNING: No score variation (std={scores_array.std():.6f})")
        print("  All TFs have the same edge score.")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70)

if __name__ == "__main__":
    main()
