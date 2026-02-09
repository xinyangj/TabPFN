#!/usr/bin/env python3
"""Test sequential_rollout strategy on DREAM4_100.

This script tests only the sequential_rollout strategy on DREAM4_100_1
to monitor its progress and performance on a larger dataset.
"""

import time
from pathlib import Path

import numpy as np

from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn,
)


def test_sequential_rollout_dream4_100():
    """Test sequential_rollout strategy on DREAM4_100_1."""
    print("=" * 80)
    print("SEQUENTIAL ROLLOUT TEST - DREAM4_100_1")
    print("=" * 80)

    # Setup
    project_root = Path(__file__).parent.parent
    dream4_path = project_root / "data" / "dream4"

    # Load dataset
    print("\nLoading DREAM4_100_1 dataset...")
    loader = DREAMChallengeLoader(data_path=str(dream4_path))
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=100,
        network_id=1
    )

    print(f"  Expression shape: {expression.shape}")
    print(f"  Genes: {len(gene_names)}")
    print(f"  TFs: {len(tf_names)}")
    print(f"  Gold standard edges: {len(gold_standard)}")

    # Preprocess
    print("\nPreprocessing data...")
    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )
    target_genes = preprocessor.get_target_names()

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Target genes: {len(target_genes)}")

    # Run sequential_rollout
    print("\n" + "=" * 80)
    print("RUNNING SEQUENTIAL ROLLOUT")
    print("=" * 80)

    start_time = time.time()
    grn = TabPFNGRNRegressor(
        tf_names,
        target_genes,
        n_estimators=1,
        edge_score_strategy="sequential_rollout",
    )

    print("\nFitting model (this will take a while for 59 target genes)...")
    grn.fit(X, y)

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds")

    # Get edge scores
    edge_scores = grn.get_edge_scores()
    print(f"\nEdge scores extracted: {len(edge_scores)} entries")

    # Print score statistics
    score_values = list(edge_scores.values())
    print(f"Score range: [{min(score_values):.6f}, {max(score_values):.6f}]")
    print(f"Mean score: {np.mean(score_values):.6f}")
    print(f"Std score: {np.std(score_values):.6f}")
    print(f"Median score: {np.median(score_values):.6f}")

    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATING RESULTS")
    print("=" * 80)

    k_values = [10, 50, 100]
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

    print(f"\nResults:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  Training time: {training_time:.2f}s")

    for k in k_values:
        print(f"  Precision@{k}:  {metrics[f'precision@{k}']:.4f}")
        print(f"  Recall@{k}:     {metrics[f'recall@{k}']:.4f}")
        print(f"  F1@{k}:         {metrics[f'f1@{k}']:.4f}")

    # Compare with self_attention
    print("\n" + "=" * 80)
    print("COMPARISON WITH SELF_ATTENTION")
    print("=" * 80)

    print(f"\nTesting self_attention...")
    start = time.time()
    grn_cmp = TabPFNGRNRegressor(
        tf_names,
        target_genes,
        n_estimators=1,
        edge_score_strategy="self_attention",
    )
    grn_cmp.fit(X, y)
    scores = grn_cmp.get_edge_scores()
    metrics_cmp = evaluate_grn(scores, gold_standard, k_values=k_values)
    elapsed = time.time() - start

    print(f"  AUPR: {metrics_cmp['aupr']:.4f}, AUROC: {metrics_cmp['auroc']:.4f}, Time: {elapsed:.2f}s")

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Strategy':<25} {'AUPR':<10} {'AUROC':<10} {'Time (s)':<10}")
    print("-" * 80)

    results = [
        ("sequential_rollout", metrics, training_time),
        ("self_attention", metrics_cmp, elapsed),
    ]

    for strat, met, t in sorted(results, key=lambda x: x[1]["aupr"], reverse=True):
        print(f"{strat:<25} {met['aupr']:<10.4f} {met['auroc']:<10.4f} {t:<10.2f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_sequential_rollout_dream4_100()
