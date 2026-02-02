"""Gene Regulatory Network (GRN) Inference Example.

This script demonstrates how to use TabPFN for GRN inference:
1. Load gene expression data
2. Preprocess for GRN analysis
3. Train TabPFN to predict target gene expression
4. Infer regulatory relationships from attention
5. Evaluate against gold standard

DREAM Challenge datasets are used for validation:
- DREAM4: Small synthetic networks (10-100 genes)
- DREAM5: Large-scale real networks (E. coli, yeast, S. aureus)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

# You can set HF_TOKEN environment variable for model download
# os.environ["HF_TOKEN"] = "your_token_here"

from tabpfn.grn import (
    AttentionHeatmapVisualizer,
    DREAMChallengeLoader,
    EdgeScoreVisualizer,
    GRNPreprocessor,
    GRNNetworkVisualizer,
    TabPFNGRNRegressor,
    create_evaluation_summary_plot,
    compute_auroc,
    compute_aupr,
    evaluate_grn,
)


def main() -> None:
    """Run GRN inference example."""
    print("=" * 70)
    print("TabPFN Gene Regulatory Network (GRN) Inference Example")
    print("=" * 70)

    # ============================================================================
    # 1. Load Dataset
    # ============================================================================
    print("\n[1] Loading DREAM4 dataset (10 genes)...")

    data_path = Path(tempfile.gettempdir()) / "dream_data"
    loader = DREAMChallengeLoader(data_path=str(data_path))

    # Load DREAM4 10-gene network (good for quick testing)
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=10, network_id=1
    )

    n_samples, n_genes = expression.shape
    n_tfs = len(tf_names)
    n_targets = len([g for g in gene_names if g not in set(tf_names)])

    print(f"  - Samples: {n_samples}")
    print(f"  - Genes: {n_genes}")
    print(f"  - Transcription Factors (TFs): {n_tfs}")
    print(f"  - Target genes: {n_targets}")
    print(f"  - Gold standard edges: {len(gold_standard)}")

    # ============================================================================
    # 2. Preprocess Data
    # ============================================================================
    print("\n[2] Preprocessing expression data...")

    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )

    target_genes = preprocessor.get_target_names()

    print(f"  - Input shape (TF expression): {X.shape}")
    print(f"  - Output shape (target expression): {y.shape}")

    # ============================================================================
    # 3. Train GRN Model
    # ============================================================================
    print("\n[3] Training TabPFN GRN model...")
    print("  Note: This uses TabPFN as a frozen foundation model with")
    print("        in-context learning (no weight updates).")

    grn_model = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=1,  # Use 1 estimator for faster inference
        device="cpu",
        attention_aggregation="mean",
    )

    grn_model.fit(X, y)

    print(f"  - Trained {len(grn_model.target_models_)} target models")
    print(f"  - Computed {len(grn_model.edge_scores_)} edge scores")

    # ============================================================================
    # 4. Infer Regulatory Network
    # ============================================================================
    print("\n[4] Inferring gene regulatory network...")

    # Get top edges by score
    edge_scores = grn_model.get_edge_scores()

    # Sort edges by score
    sorted_edges = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)

    print(f"  - Top 5 predicted edges:")
    for (tf, target), score in sorted_edges[:5]:
        is_true = "✓" if (tf, target) in gold_standard else " "
        print(f"    {is_true} {tf} -> {target}: {score:.4f}")

    # ============================================================================
    # 5. Evaluate Against Gold Standard
    # ============================================================================
    print("\n[5] Evaluating inferred network...")

    metrics = evaluate_grn(edge_scores, gold_standard, k_values=[5, 10, 20])

    print("  - Evaluation metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"    {metric_name}: {metric_value:.4f}")

    # ============================================================================
    # 6. Visualize Results
    # ============================================================================
    print("\n[6] Creating visualizations...")

    output_dir = Path(tempfile.gettempdir()) / "grn_results"
    output_dir.mkdir(exist_ok=True)

    # Network visualization
    network_viz = GRNNetworkVisualizer()
    grn = grn_model.infer_grn(top_k=20)
    network_viz.plot_network(
        grn,
        output_path=output_dir / "network.pdf",
        max_nodes=50,
    )
    print(f"  - Saved network plot to: {output_dir / 'network.pdf'}")

    # Edge score distribution
    score_viz = EdgeScoreVisualizer()
    score_viz.plot_score_distribution(
        edge_scores,
        gold_standard,
        output_path=output_dir / "score_distribution.pdf",
    )
    print(f"  - Saved score distribution to: {output_dir / 'score_distribution.pdf'}")

    # Precision-Recall curve
    score_viz.plot_precision_recall_curve(
        edge_scores,
        gold_standard,
        output_path=output_dir / "pr_curve.pdf",
    )
    print(f"  - Saved PR curve to: {output_dir / 'pr_curve.pdf'}")

    # ROC curve
    score_viz.plot_roc_curve(
        edge_scores,
        gold_standard,
        output_path=output_dir / "roc_curve.pdf",
    )
    print(f"  - Saved ROC curve to: {output_dir / 'roc_curve.pdf'}")

    # Evaluation summary
    create_evaluation_summary_plot(
        metrics,
        output_path=output_dir / "evaluation_summary.pdf",
    )
    print(f"  - Saved evaluation summary to: {output_dir / 'evaluation_summary.pdf'}")

    # ============================================================================
    # 7. Summary
    # ============================================================================
    print("\n" + "=" * 70)
    print("GRN Inference Complete!")
    print("=" * 70)
    print(f"\nKey Results:")
    print(f"  - AUPR: {metrics['aupr']:.4f} (Area Under Precision-Recall)")
    print(f"  - AUROC: {metrics['auroc']:.4f} (Area Under ROC)")
    print(f"  - Precision@5: {metrics['precision@5']:.4f}")
    print(f"  - Recall@5: {metrics['recall@5']:.4f}")
    print(f"\nOutput files saved to: {output_dir}")

    # ============================================================================
    # 8. Advanced: Attention Visualization (if available)
    # ============================================================================
    print("\n[8] (Optional) Visualizing attention patterns...")

    if hasattr(grn_model, "attention_weights_") and grn_model.attention_weights_:
        # Get attention for first target
        first_target = target_genes[0]
        if first_target in grn_model.attention_weights_:
            attention = grn_model.attention_weights_[first_target]

            attn_viz = AttentionHeatmapVisualizer()
            attn_viz.plot_multi_layer_attention(
                attention,
                tf_names=tf_names,
                output_path=output_dir / "attention_heatmap.pdf",
            )
            print(f"  - Saved attention heatmap to: {output_dir / 'attention_heatmap.pdf'}")
    else:
        print("  - Skip: Attention weights not available")

    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
