#!/usr/bin/env python
"""Comprehensive validation of GRN inference on DREAM4 datasets."""

import numpy as np
from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn,
)

def test_single_network(network_size, network_id):
    """Test GRN inference on a single network."""
    loader = DREAMChallengeLoader()
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=network_size, network_id=network_id
    )

    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )
    target_genes = preprocessor.get_target_names()

    grn_model = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,
        attention_aggregation="mean",
        device="cpu",
    )
    grn_model.fit(X, y)

    edge_scores = grn_model.get_edge_scores()
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=[5, 10, 20])

    return metrics

def main():
    print("="*70)
    print("Comprehensive GRN Validation on DREAM4 Datasets")
    print("="*70)

    results = []

    # Test DREAM4-10 networks
    print("\n" + "="*70)
    print("DREAM4-10 Networks (5 networks)")
    print("="*70)
    for network_id in range(1, 6):
        print(f"\nNetwork {network_id}:")
        try:
            metrics = test_single_network(10, network_id)
            results.append(("DREAM4-10", network_id, metrics))
            print(f"  AUPR:  {metrics['aupr']:.4f}")
            print(f"  AUROC: {metrics['auroc']:.4f}")
            print(f"  Precision@10: {metrics.get('precision@10', 0):.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    # Test DREAM4-100
    print("\n" + "="*70)
    print("DREAM4-100 Network")
    print("="*70)
    print("\nNetwork 1:")
    try:
        metrics = test_single_network(100, 1)
        results.append(("DREAM4-100", 1, metrics))
        print(f"  AUPR:  {metrics['aupr']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")
        print(f"  Precision@100: {metrics.get('precision@100', 0):.4f}")
    except Exception as e:
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)

    # Group by dataset
    dream4_10_results = [r for r in results if r[0] == "DREAM4-10"]
    dream4_100_results = [r for r in results if r[0] == "DREAM4-100"]

    if dream4_10_results:
        auprs = [r[2]['aupr'] for r in dream4_10_results]
        aurocs = [r[2]['auroc'] for r in dream4_10_results]
        print(f"\nDREAM4-10 ({len(dream4_10_results)} networks):")
        print(f"  AUPR:  mean={np.mean(auprs):.4f}, std={np.std(auprs):.4f}")
        print(f"  AUROC: mean={np.mean(aurocs):.4f}, std={np.std(aurocs):.4f}")

    if dream4_100_results:
        auprs = [r[2]['aupr'] for r in dream4_100_results]
        aurocs = [r[2]['auroc'] for r in dream4_100_results]
        print(f"\nDREAM4-100 ({len(dream4_100_results)} network):")
        print(f"  AUPR:  {auprs[0]:.4f}")
        print(f"  AUROC: {aurocs[0]:.4f}")

    # Overall validation
    print("\n" + "="*70)
    print("Validation Results")
    print("="*70)

    all_auprs = [r[2]['aupr'] for r in results]
    all_aurocs = [r[2]['auroc'] for r in results]

    mean_aupr = np.mean(all_auprs)
    mean_auroc = np.mean(all_aurocs)

    print(f"\nOverall Performance:")
    print(f"  Mean AUPR:  {mean_aupr:.4f}")
    print(f"  Mean AUROC: {mean_auroc:.4f}")

    if mean_aupr > 0.15:
        print(f"\n✓ SUCCESS: Mean AUPR ({mean_aupr:.4f}) exceeds minimum threshold (0.15)")
    else:
        print(f"\n✗ WARNING: Mean AUPR ({mean_aupr:.4f}) below minimum threshold (0.15)")

    if mean_auroc > 0.6:
        print(f"✓ SUCCESS: Mean AUROC ({mean_auroc:.4f}) exceeds minimum threshold (0.6)")
    else:
        print(f"✗ WARNING: Mean AUROC ({mean_auroc:.4f}) below minimum threshold (0.6)")

    print("\n" + "="*70)
    print("Validation Complete!")
    print("="*70)

if __name__ == "__main__":
    main()
