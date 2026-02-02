#!/usr/bin/env python
"""Explore different edge score extraction strategies for GRN inference.

This script compares 4 different strategies for extracting TF-target edge scores
from TabPFN's attention weights:
1. self_attention: Use diagonal feat_attn[tf_idx, tf_idx]
2. tf_to_target: Use feat_attn[tf_idx, -1] (TF attends to target)
3. target_to_tf: Use feat_attn[-1, tf_idx] (Target attends to TF)
4. combined: Weighted average of all three

The key insight is that TabPFN concatenates the target as the last feature position,
making these alternative strategies possible.
"""

import numpy as np
from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn,
)


def test_strategies_on_network(network_size: int = 10, network_id: int = 1):
    """Test all edge score strategies on a single DREAM4 network.

    Parameters
    ----------
    network_size : int
        Size of the network (10 or 100)
    network_id : int
        Network ID (1-5 for size 10, 1 for size 100)

    Returns
    -------
    results : dict
        Dictionary mapping strategy names to evaluation metrics
    """
    # Load data
    loader = DREAMChallengeLoader()
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=network_size, network_id=network_id
    )

    # Preprocess
    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )
    target_genes = preprocessor.get_target_names()

    # Define strategies to test
    strategies = ["self_attention", "tf_to_target", "target_to_tf", "combined"]
    results = {}

    print(f"\nTesting on DREAM4-{network_size} Network {network_id}")
    print(f"  {len(tf_names)} TFs, {len(target_genes)} targets, {len(gold_standard)} gold edges")

    for strategy in strategies:
        print(f"\n  Strategy: {strategy}")

        # Train GRN model with this strategy
        grn_model = TabPFNGRNRegressor(
            tf_names=tf_names,
            target_genes=target_genes,
            n_estimators=1,  # Use 1 for speed
            edge_score_strategy=strategy,
            device="cpu",
        )
        grn_model.fit(X, y)

        # Get edge scores
        edge_scores = grn_model.get_edge_scores()

        # Evaluate
        metrics = evaluate_grn(edge_scores, gold_standard, k_values=[5, 10, 20])
        results[strategy] = metrics

        print(f"    AUPR:  {metrics['aupr']:.4f}")
        print(f"    AUROC: {metrics['auroc']:.4f}")

        # Show top edges for this strategy
        sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
        correct = sum(1 for (tf, tgt), _ in sorted_edges[:10] if (tf, tgt) in gold_standard)
        print(f"    Precision@10: {correct}/10")

    return results


def main():
    """Run comprehensive comparison of edge score strategies."""
    print("=" * 80)
    print("Edge Score Strategy Exploration for TabPFN GRN Inference")
    print("=" * 80)

    all_results = {}

    # Test on multiple DREAM4-10 networks
    print("\n" + "=" * 80)
    print("Testing on DREAM4-10 (Small Synthetic Networks)")
    print("=" * 80)

    dream4_10_results = []
    for network_id in range(1, 6):
        try:
            results = test_strategies_on_network(network_size=10, network_id=network_id)
            dream4_10_results.append(results)
        except Exception as e:
            print(f"  ERROR on network {network_id}: {e}")

    # Aggregate results
    if dream4_10_results:
        print("\n" + "=" * 80)
        print("Summary: DREAM4-10 (5 networks)")
        print("=" * 80)

        strategies = ["self_attention", "tf_to_target", "target_to_tf", "combined"]
        for strategy in strategies:
            auprs = [r[strategy]['aupr'] for r in dream4_10_results if strategy in r]
            aurocs = [r[strategy]['auroc'] for r in dream4_10_results if strategy in r]
            if auprs:
                print(f"\n  {strategy}:")
                print(f"    AUPR:  mean={np.mean(auprs):.4f}, std={np.std(auprs):.4f}")
                print(f"    AUROC: mean={np.mean(aurocs):.4f}, std={np.std(aurocs):.4f}")

    # Test on DREAM4-100
    print("\n" + "=" * 80)
    print("Testing on DREAM4-100 (Larger Synthetic Network)")
    print("=" * 80)

    try:
        results = test_strategies_on_network(network_size=100, network_id=1)
        all_results['DREAM4-100'] = results

        print("\n" + "=" * 80)
        print("Summary: DREAM4-100")
        print("=" * 80)
        for strategy, metrics in results.items():
            print(f"\n  {strategy}:")
            print(f"    AUPR:  {metrics['aupr']:.4f}")
            print(f"    AUROC: {metrics['auroc']:.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)

    if dream4_10_results:
        print("\nMean AUPR across DREAM4-10 networks:")
        strategies = ["self_attention", "tf_to_target", "target_to_tf", "combined"]
        best_strategy = None
        best_aupr = -1

        for strategy in strategies:
            auprs = [r[strategy]['aupr'] for r in dream4_10_results if strategy in r]
            if auprs:
                mean_aupr = np.mean(auprs)
                print(f"  {strategy:20s}: {mean_aupr:.4f}")
                if mean_aupr > best_aupr:
                    best_aupr = mean_aupr
                    best_strategy = strategy

        print(f"\n  BEST: {best_strategy} (AUPR = {best_aupr:.4f})")

        # Recommendation
        print("\nRecommendation:")
        if best_strategy == "self_attention":
            print("  Use 'self_attention' (current default)")
            print("  TF self-attention (diagonal) works best as a proxy for TF importance")
        elif best_strategy == "tf_to_target":
            print("  Use 'tf_to_target' (TF attends to target)")
            print("  This directly captures TF->Target regulatory relationships")
        elif best_strategy == "target_to_tf":
            print("  Use 'target_to_tf' (Target attends to TF)")
            print("  This captures the target's dependence on each TF")
        elif best_strategy == "combined":
            print("  Use 'combined' (weighted average)")
            print("  Combining multiple attention patterns gives the best results")

    print("\n" + "=" * 80)
    print("Exploration Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
