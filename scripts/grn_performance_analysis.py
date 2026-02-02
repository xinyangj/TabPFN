"""Comprehensive GRN Performance Analysis Script.

Runs TabPFN GRN inference on DREAM datasets and compares against baseline methods.
Generates a detailed performance report with metrics and visualizations.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn,
    GRNNetworkVisualizer,
    EdgeScoreVisualizer,
    create_evaluation_summary_plot,
)


def run_tabpfnn_analysis(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
    n_estimators: int = 2,
    edge_score_strategy: str = "self_attention",
) -> dict[str, Any]:
    """Run TabPFN GRN analysis on a dataset.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix (samples x genes)
    gene_names : list[str]
        All gene names
    tf_names : list[str]
        Transcription factor names
    gold_standard : set[tuple[str, str]]
        True regulatory edges
    target_genes : list[str]
        Target gene names
    dataset_name : str
        Name of the dataset
    n_estimators : int
        Number of TabPFN estimators
    edge_score_strategy : str
        Edge score extraction strategy to use

    Returns
    -------
    dict with results
    """
    strategy_label = edge_score_strategy.replace("_", " ").title()
    print(f"\n{'='*70}")
    print(f"TabPFN GRN Analysis: {dataset_name} ({strategy_label})")
    print(f"{'='*70}")

    # Preprocess
    print("Preprocessing data...")
    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )

    # Train
    print(f"Training TabPFN GRN model (n_estimators={n_estimators})...")
    print(f"  Strategy: {edge_score_strategy}")
    start_time = time.time()

    grn_model = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=n_estimators,
        device="cpu",  # Use "cuda" if available
        attention_aggregation="mean",
        edge_score_strategy=edge_score_strategy,
    )
    grn_model.fit(X, y)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Get edge scores
    edge_scores = grn_model.get_edge_scores()

    # Evaluate
    k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

    # Add training time
    metrics["training_time"] = training_time
    metrics["num_edges_predicted"] = len(edge_scores)
    metrics["num_true_edges"] = len(gold_standard)

    # Print results
    print(f"\nResults:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    for k in k_values:
        if f"precision@{k}" in metrics:
            print(f"  Precision@{k}:  {metrics[f'precision@{k}']:.4f}")
        if f"recall@{k}" in metrics:
            print(f"  Recall@{k}:     {metrics[f'recall@{k}']:.4f}")

    return {
        "dataset": dataset_name,
        "method": f"TabPFN ({strategy_label})",
        "metrics": metrics,
        "edge_scores": edge_scores,
        "grn_model": grn_model,
        "strategy": edge_score_strategy,
    }


def run_tabpfnn_analysis_with_y(
    X: np.ndarray,
    y: np.ndarray,
    tf_names: list[str],
    target_genes: list[str],
    gold_standard: set[tuple[str, str]],
    dataset_name: str,
    n_estimators: int = 2,
    edge_score_strategy: str = "self_attention",
) -> dict[str, Any]:
    """Run TabPFN GRN analysis with pre-computed X and y matrices.

    Parameters
    ----------
    X : np.ndarray
        TF expression matrix (samples x TFs)
    y : np.ndarray
        Target expression matrix (samples x targets)
    tf_names : list[str]
        Transcription factor names
    target_genes : list[str]
        Target gene names
    gold_standard : set[tuple[str, str]]
        True regulatory edges
    dataset_name : str
        Name of the dataset
    n_estimators : int
        Number of TabPFN estimators
    edge_score_strategy : str
        Edge score extraction strategy to use

    Returns
    -------
    dict with results
    """
    strategy_label = edge_score_strategy.replace("_", " ").title()
    print(f"\n{'='*70}")
    print(f"TabPFN GRN Analysis: {dataset_name} ({strategy_label})")
    print(f"{'='*70}")

    # Train
    print(f"Training TabPFN GRN model (n_estimators={n_estimators})...")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  TFs: {len(tf_names)}, Targets: {len(target_genes)}")
    print(f"  Strategy: {edge_score_strategy}")

    start_time = time.time()

    grn_model = TabPFNGRNRegressor(
        tf_names=tf_names,
        target_genes=target_genes,
        n_estimators=n_estimators,
        device="cpu",
        attention_aggregation="mean",
        edge_score_strategy=edge_score_strategy,
    )
    grn_model.fit(X, y)

    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Get edge scores
    edge_scores = grn_model.get_edge_scores()

    # Evaluate
    k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

    # Add training time
    metrics["training_time"] = training_time
    metrics["num_edges_predicted"] = len(edge_scores)
    metrics["num_true_edges"] = len(gold_standard)

    # Print results
    print(f"\nResults:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    for k in k_values:
        if f"precision@{k}" in metrics:
            print(f"  Precision@{k}:  {metrics[f'precision@{k}']:.4f}")
        if f"recall@{k}" in metrics:
            print(f"  Recall@{k}:     {metrics[f'recall@{k}']:.4f}")

    return {
        "dataset": dataset_name,
        "method": f"TabPFN ({strategy_label})",
        "metrics": metrics,
        "edge_scores": edge_scores,
        "grn_model": grn_model,
        "strategy": edge_score_strategy,
    }


def run_correlation_baseline(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Run simple Pearson correlation baseline.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix
    gene_names : list[str]
        All gene names
    tf_names : list[str]
        Transcription factor names
    gold_standard : set[tuple[str, str]]
        True regulatory edges
    target_genes : list[str]
        Target gene names
    dataset_name : str
        Name of the dataset

    Returns
    -------
    dict with results
    """
    print(f"\n{'='*70}")
    print(f"Correlation Baseline: {dataset_name}")
    print(f"{'='*70}")

    # Compute correlation between TFs and targets
    tf_indices = [gene_names.index(tf) for tf in tf_names]
    target_indices = [gene_names.index(tgt) for tgt in target_genes]

    edge_scores = {}
    for i, tf in enumerate(tf_names):
        for j, tgt in enumerate(target_genes):
            corr = np.corrcoef(
                expression[:, tf_indices[i]],
                expression[:, target_indices[j]]
            )[0, 1]
            # Use absolute correlation as score
            edge_scores[(tf, tgt)] = abs(corr) if not np.isnan(corr) else 0.0

    # Evaluate
    k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

    # Add metadata
    metrics["training_time"] = 0.0
    metrics["num_edges_predicted"] = len(edge_scores)
    metrics["num_true_edges"] = len(gold_standard)

    # Print results
    print(f"\nResults:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")

    return {
        "dataset": dataset_name,
        "method": "Correlation",
        "metrics": metrics,
        "edge_scores": edge_scores,
    }


def run_genie3_baseline(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Run GENIE3 baseline.

    First tries to use arboreto.algo.genie3, with sklearn fallback.
    GENIE3 uses Random Forest regression to predict each target gene from TFs.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix (samples x genes)
    gene_names : list[str]
        All gene names
    tf_names : list[str]
        Transcription factor names
    gold_standard : set[tuple[str, str]]
        True regulatory edges
    target_genes : list[str]
        Target gene names
    dataset_name : str
        Name of the dataset

    Returns
    -------
    dict with results
    """
    print(f"\n{'='*70}")
    print(f"GENIE3 Baseline: {dataset_name}")
    print(f"{'='*70}")

    # First try arboreto
    try:
        import pandas as pd
        from arboreto.algo import genie3

        # Convert to DataFrame for arboreto (genes as columns, samples as rows)
        expression_df = pd.DataFrame(expression, columns=gene_names)

        print(f"  Running GENIE3 (arboreto) with {len(tf_names)} TFs and {len(gene_names)} genes...")

        start_time = time.time()

        # Use arboreto's GENIE3 implementation
        # Returns DataFrame with columns: TF, target, importance
        network = genie3(
            expression_data=expression_df,
            tf_names=tf_names,
            client_or_address='local',
            seed=42,
        )

        training_time = time.time() - start_time
        print(f"  GENIE3 completed in {training_time:.2f} seconds")

        # Convert network DataFrame to edge_scores dict
        edge_scores = {}
        for _, row in network.iterrows():
            tf = row['TF']
            target = row['target']
            importance = row['importance']
            # Only include edges where target is in our target_genes list
            if target in target_genes:
                edge_scores[(tf, target)] = importance

        # Evaluate
        k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
        metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

        # Add metadata
        metrics["training_time"] = training_time
        metrics["num_edges_predicted"] = len(edge_scores)
        metrics["num_true_edges"] = len(gold_standard)

        # Print results
        print(f"\nResults:")
        print(f"  AUPR:  {metrics['aupr']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")

        return {
            "dataset": dataset_name,
            "method": "GENIE3",
            "metrics": metrics,
            "edge_scores": edge_scores,
        }

    except ImportError:
        print("  arboreto not installed, trying sklearn fallback...")
        return _run_genie3_sklearn(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )
    except Exception as e:
        print(f"  arboreto failed: {e}")
        print("  Trying sklearn fallback...")
        return _run_genie3_sklearn(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )


def _run_genie3_sklearn(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Run GENIE3 baseline using sklearn RandomForest (fallback).

    This is a sklearn-based implementation of GENIE3 when arboreto is unavailable.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor

        print(f"  Running GENIE3 (sklearn) with {len(tf_names)} TFs and {len(gene_names)} genes...")

        start_time = time.time()

        # Create mapping from gene names to column indices
        gene_to_idx = {name: i for i, name in enumerate(gene_names)}
        tf_indices = [gene_to_idx[tf] for tf in tf_names if tf in gene_to_idx]

        edge_scores = {}

        # Train a Random Forest for each target gene
        for target_name in target_genes:
            if target_name not in gene_to_idx:
                continue

            target_idx = gene_to_idx[target_name]
            y = expression[:, target_idx]
            X = expression[:, tf_indices]

            # Train Random Forest
            rf = RandomForestRegressor(
                n_estimators=100,
                max_features='sqrt',
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X, y)

            # Get feature importances as edge scores
            for i, tf_name in enumerate(tf_names):
                if i < len(tf_indices):
                    edge_scores[(tf_name, target_name)] = rf.feature_importances_[i]

        training_time = time.time() - start_time
        print(f"  GENIE3 (sklearn) completed in {training_time:.2f} seconds")

        # Evaluate
        k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
        metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

        # Add metadata
        metrics["training_time"] = training_time
        metrics["num_edges_predicted"] = len(edge_scores)
        metrics["num_true_edges"] = len(gold_standard)

        # Print results
        print(f"\nResults:")
        print(f"  AUPR:  {metrics['aupr']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")

        return {
            "dataset": dataset_name,
            "method": "GENIE3 (sklearn)",
            "metrics": metrics,
            "edge_scores": edge_scores,
        }

    except Exception as e:
        print(f"  GENIE3 (sklearn) failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_grnboost2_baseline(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Run GRNBoost2 baseline.

    First tries to use arboreto.algo.grnboost2, with sklearn fallback.
    GRNBoost2 uses Gradient Boosting to predict each target gene from TFs.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix (samples x genes)
    gene_names : list[str]
        All gene names
    tf_names : list[str]
        Transcription factor names
    gold_standard : set[tuple[str, str]]
        True regulatory edges
    target_genes : list[str]
        Target gene names
    dataset_name : str
        Name of the dataset

    Returns
    -------
    dict with results
    """
    print(f"\n{'='*70}")
    print(f"GRNBoost2 Baseline: {dataset_name}")
    print(f"{'='*70}")

    # First try arboreto
    try:
        import pandas as pd
        from arboreto.algo import grnboost2

        # Convert to DataFrame for arboreto (genes as columns, samples as rows)
        expression_df = pd.DataFrame(expression, columns=gene_names)

        print(f"  Running GRNBoost2 (arboreto) with {len(tf_names)} TFs and {len(gene_names)} genes...")

        start_time = time.time()

        # Use arboreto's GRNBoost2 implementation
        # Returns DataFrame with columns: TF, target, importance
        network = grnboost2(
            expression_data=expression_df,
            tf_names=tf_names,
            client_or_address='local',
            seed=42,
        )

        training_time = time.time() - start_time
        print(f"  GRNBoost2 completed in {training_time:.2f} seconds")

        # Convert network DataFrame to edge_scores dict
        edge_scores = {}
        for _, row in network.iterrows():
            tf = row['TF']
            target = row['target']
            importance = row['importance']
            # Only include edges where target is in our target_genes list
            if target in target_genes:
                edge_scores[(tf, target)] = importance

        # Evaluate
        k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
        metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

        # Add metadata
        metrics["training_time"] = training_time
        metrics["num_edges_predicted"] = len(edge_scores)
        metrics["num_true_edges"] = len(gold_standard)

        # Print results
        print(f"\nResults:")
        print(f"  AUPR:  {metrics['aupr']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")

        return {
            "dataset": dataset_name,
            "method": "GRNBoost2",
            "metrics": metrics,
            "edge_scores": edge_scores,
        }

    except ImportError:
        print("  arboreto not installed, trying sklearn fallback...")
        return _run_grnboost2_sklearn(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )
    except Exception as e:
        print(f"  arboreto failed: {e}")
        print("  Trying sklearn fallback...")
        return _run_grnboost2_sklearn(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )


def _run_grnboost2_sklearn(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Run GRNBoost2 baseline using sklearn GradientBoosting (fallback).

    This is a sklearn-based implementation of GRNBoost2 when arboreto is unavailable.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor

        print(f"  Running GRNBoost2 (sklearn) with {len(tf_names)} TFs and {len(gene_names)} genes...")

        start_time = time.time()

        # Create mapping from gene names to column indices
        gene_to_idx = {name: i for i, name in enumerate(gene_names)}
        tf_indices = [gene_to_idx[tf] for tf in tf_names if tf in gene_to_idx]

        edge_scores = {}

        # Train a Gradient Boosting model for each target gene
        for target_name in target_genes:
            if target_name not in gene_to_idx:
                continue

            target_idx = gene_to_idx[target_name]
            y = expression[:, target_idx]
            X = expression[:, tf_indices]

            # Train Gradient Boosting
            gb = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                min_samples_leaf=1,
                random_state=42,
            )
            gb.fit(X, y)

            # Get feature importances as edge scores
            for i, tf_name in enumerate(tf_names):
                if i < len(tf_indices):
                    edge_scores[(tf_name, target_name)] = gb.feature_importances_[i]

        training_time = time.time() - start_time
        print(f"  GRNBoost2 (sklearn) completed in {training_time:.2f} seconds")

        # Evaluate
        k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
        metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

        # Add metadata
        metrics["training_time"] = training_time
        metrics["num_edges_predicted"] = len(edge_scores)
        metrics["num_true_edges"] = len(gold_standard)

        # Print results
        print(f"\nResults:")
        print(f"  AUPR:  {metrics['aupr']:.4f}")
        print(f"  AUROC: {metrics['auroc']:.4f}")

        return {
            "dataset": dataset_name,
            "method": "GRNBoost2 (sklearn)",
            "metrics": metrics,
            "edge_scores": edge_scores,
        }

    except Exception as e:
        print(f"  GRNBoost2 (sklearn) failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_mutual_information_baseline(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: set[tuple[str, str]],
    target_genes: list[str],
    dataset_name: str,
) -> dict[str, Any]:
    """Run Mutual Information baseline.

    Parameters
    ----------
    expression : np.ndarray
        Gene expression matrix
    gene_names : list[str]
        All gene names
    tf_names : list[str]
        Transcription factor names
    gold_standard : set[tuple[str, str]]
        True regulatory edges
    target_genes : list[str]
        Target gene names
    dataset_name : str
        Name of the dataset

    Returns
    -------
    dict with results
    """
    print(f"\n{'='*70}")
    print(f"Mutual Information Baseline: {dataset_name}")
    print(f"{'='*70}")

    from sklearn.feature_selection import mutual_info_regression

    # Discretize expression for MI estimation
    tf_indices = [gene_names.index(tf) for tf in tf_names]
    target_indices = [gene_names.index(tgt) for tgt in target_genes]

    edge_scores = {}
    for j, tgt in enumerate(target_genes):
        # Compute MI between all TFs and this target
        mi_scores = mutual_info_regression(
            expression[:, tf_indices],
            expression[:, target_indices[j]],
            random_state=42
        )
        for i, tf in enumerate(tf_names):
            edge_scores[(tf, tgt)] = mi_scores[i]

    # Normalize scores
    max_score = max(edge_scores.values()) if edge_scores else 1.0
    edge_scores = {k: v / max_score for k, v in edge_scores.items()}

    # Evaluate
    k_values = [10, 50, 100] if len(gold_standard) > 50 else [5, 10, 20]
    metrics = evaluate_grn(edge_scores, gold_standard, k_values=k_values)

    # Add metadata
    metrics["training_time"] = 0.0
    metrics["num_edges_predicted"] = len(edge_scores)
    metrics["num_true_edges"] = len(gold_standard)

    # Print results
    print(f"\nResults:")
    print(f"  AUPR:  {metrics['aupr']:.4f}")
    print(f"  AUROC: {metrics['auroc']:.4f}")

    return {
        "dataset": dataset_name,
        "method": "Mutual Information",
        "metrics": metrics,
        "edge_scores": edge_scores,
    }


def generate_comparison_report(
    results: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Generate comparison report and visualizations.

    Parameters
    ----------
    results : list of dict
        All results from different methods
    output_dir : Path
        Directory to save outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset
    datasets = {}
    for r in results:
        dataset = r["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(r)

    # Generate report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("GRN INFERENCE PERFORMANCE REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    for dataset_name, dataset_results in datasets.items():
        report_lines.append(f"\n{'='*80}")
        report_lines.append(f"Dataset: {dataset_name}")
        report_lines.append(f"{'='*80}")
        report_lines.append("")

        # Create table header
        report_lines.append(f"{'Method':<25} {'AUPR':<10} {'AUROC':<10} {'Time (s)':<10}")
        report_lines.append("-" * 80)

        # Sort by AUPR
        sorted_results = sorted(
            dataset_results,
            key=lambda x: x["metrics"]["aupr"],
            reverse=True
        )

        for r in sorted_results:
            method = r["method"]
            aupr = r["metrics"]["aupr"]
            auroc = r["metrics"]["auroc"]
            time_s = r["metrics"]["training_time"]
            report_lines.append(f"{method:<25} {aupr:<10.4f} {auroc:<10.4f} {time_s:<10.2f}")

        report_lines.append("")

        # Add precision@k values
        k_values = sorted(set(
            k for r in dataset_results
            for k in r["metrics"].keys()
            if "@" in k and "precision@" in k
        ))

        if k_values:
            for k in k_values:
                k_val = k.split("@")[1]
                report_lines.append(f"\nPrecision@{k_val}:")
                for r in sorted_results:
                    if k in r["metrics"]:
                        report_lines.append(f"  {r['method']:<25} {r['metrics'][k]:.4f}")

        report_lines.append("")

        # Generate comparison plots for this dataset
        if len(dataset_results) > 1:
            # Get TabPFN result
            tabpfn_result = next((r for r in dataset_results if r["method"] == "TabPFN"), None)

            # Create comparison plot
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # Bar chart comparison
            methods = [r["method"] for r in sorted_results]
            auprs = [r["metrics"]["aupr"] for r in sorted_results]
            aurocs = [r["metrics"]["auroc"] for r in sorted_results]

            x = np.arange(len(methods))
            width = 0.35

            axes[0].bar(x - width/2, auprs, width, label="AUPR", color="steelblue")
            axes[0].bar(x + width/2, aurocs, width, label="AUROC", color="coral")
            axes[0].set_ylabel("Score")
            axes[0].set_title(f"{dataset_name}: Metric Comparison")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(methods, rotation=15, ha="right", fontsize=9)
            axes[0].legend()
            axes[0].set_ylim([0, 1])

            # Precision@K comparison (plot all methods that have precision@k data)
            if tabpfn_result:
                k_metrics = [k for k in tabpfn_result["metrics"].keys() if "precision@" in k]
                if k_metrics:
                    k_vals = sorted([int(k.split("@")[1]) for k in k_metrics])

                    # Create line plot for each method
                    colors = plt.cm.tab10(np.linspace(0, 1, len(dataset_results)))
                    for i, r in enumerate(dataset_results):
                        precs = [r["metrics"].get(f"precision@{k}", 0) for k in k_vals]
                        line_style = "-" if r["method"] == "TabPFN" else "--"
                        axes[1].plot(k_vals, precs, line_style, label=r["method"],
                                   linewidth=2 if r["method"] == "TabPFN" else 1.5,
                                   color=colors[i], markersize=6)

                    axes[1].set_xlabel("K")
                    axes[1].set_ylabel("Precision")
                    axes[1].set_title(f"{dataset_name}: Precision@K")
                    axes[1].legend(fontsize=8, ncol=2)
                    axes[1].grid(True, alpha=0.3)
                    axes[1].set_ylim([0, 1])

            plt.tight_layout()
            plt.savefig(output_dir / f"{dataset_name}_comparison.png", dpi=150)
            plt.close()

    # Save report
    report_path = output_dir / "GRN_Performance_Report.txt"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # Print report
    print("\n" + "\n".join(report_lines))

    # Save JSON summary
    summary = {}
    for dataset_name, dataset_results in datasets.items():
        summary[dataset_name] = {}
        for r in dataset_results:
            summary[dataset_name][r["method"]] = {
                "aupr": r["metrics"]["aupr"],
                "auroc": r["metrics"]["auroc"],
                "training_time": r["metrics"]["training_time"],
            }
            # Add precision@k values
            for k, v in r["metrics"].items():
                if "@" in k:
                    summary[dataset_name][r["method"]][k] = v

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print(f"{'='*80}")


def main() -> None:
    """Run comprehensive GRN performance analysis."""
    print("=" * 80)
    print("GRN PERFORMANCE ANALYSIS - TabPFN vs Baselines")
    print("=" * 80)

    # Setup - use local data directory
    # DREAM4 data is in data/dream4/dream4/ (nested structure)
    # DREAM5 data is in data/dream5/
    project_root = Path(__file__).parent.parent
    dream4_path = project_root / "data" / "dream4"
    dream5_path = project_root / "data" / "dream5"
    output_dir = project_root / "results" / "grn_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Define TabPFN edge score strategies to test
    tabpfn_strategies = [
        "self_attention",
        "tf_to_target",
        "target_to_tf",
        "combined",
        "combined_best",
    ]

    # ============================================================================
    # DREAM4 Analysis (10 genes - fast for testing)
    # ============================================================================
    print("\n" + "="*80)
    print("DREAM4 ANALYSIS (10 genes, 5 networks)")
    print("="*80)

    # Create loader for DREAM4 data
    dream4_loader = DREAMChallengeLoader(data_path=str(dream4_path))

    for network_id in range(1, 3):  # Test 2 networks for speed
        expression, gene_names, tf_names, gold_standard = dream4_loader.load_dream4(
            network_size=10,
            network_id=network_id
        )

        preprocessor = GRNPreprocessor(normalization="zscore")
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression, gene_names, tf_names
        )
        target_genes = preprocessor.get_target_names()

        dataset_name = f"DREAM4_10_{network_id}"

        # Run TabPFN with all strategies
        for strategy in tabpfn_strategies:
            result_tabpfn = run_tabpfnn_analysis_with_y(
                X, y, tf_names, target_genes, gold_standard,
                dataset_name, n_estimators=1, edge_score_strategy=strategy
            )
            all_results.append(result_tabpfn)

        # Run correlation baseline
        result_corr = run_correlation_baseline(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )
        all_results.append(result_corr)

        # Run MI baseline
        result_mi = run_mutual_information_baseline(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )
        all_results.append(result_mi)

        # Run GENIE3 baseline
        result_genie3 = run_genie3_baseline(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )
        if result_genie3:
            all_results.append(result_genie3)

        # Run GRNBoost2 baseline
        result_grnboost2 = run_grnboost2_baseline(
            expression, gene_names, tf_names, gold_standard,
            target_genes, dataset_name
        )
        if result_grnboost2:
            all_results.append(result_grnboost2)

    # ============================================================================
    # DREAM4 Analysis (100 genes - larger test)
    # ============================================================================
    print("\n" + "="*80)
    print("DREAM4 ANALYSIS (100 genes)")
    print("="*80)

    expression, gene_names, tf_names, gold_standard = dream4_loader.load_dream4(
        network_size=100,
        network_id=1
    )

    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression, gene_names, tf_names
    )
    target_genes = preprocessor.get_target_names()

    dataset_name = "DREAM4_100_1"

    # Run TabPFN with all strategies
    for strategy in tabpfn_strategies:
        result_tabpfn = run_tabpfnn_analysis_with_y(
            X, y, tf_names, target_genes, gold_standard,
            dataset_name, n_estimators=1, edge_score_strategy=strategy
        )
        all_results.append(result_tabpfn)

    # Run correlation baseline
    result_corr = run_correlation_baseline(
        expression, gene_names, tf_names, gold_standard,
        target_genes, dataset_name
    )
    all_results.append(result_corr)

    # Run GENIE3 baseline
    result_genie3 = run_genie3_baseline(
        expression, gene_names, tf_names, gold_standard,
        target_genes, dataset_name
    )
    if result_genie3:
        all_results.append(result_genie3)

    # Run GRNBoost2 baseline
    result_grnboost2 = run_grnboost2_baseline(
        expression, gene_names, tf_names, gold_standard,
        target_genes, dataset_name
    )
    if result_grnboost2:
        all_results.append(result_grnboost2)

    # ============================================================================
    # DREAM5 E. coli Analysis (real data)
    # ============================================================================
    print("\n" + "="*80)
    print("DREAM5 E. COLI ANALYSIS (real data)")
    print("="*80)

    try:
        import pandas as pd

        # Create loader for DREAM5 data
        dream5_loader = DREAMChallengeLoader(data_path=str(dream5_path))

        expression, gene_names, tf_names, gold_standard = dream5_loader.load_dream5_ecoli()

        # Convert gold_standard DataFrame to set if needed
        if isinstance(gold_standard, pd.DataFrame):
            gs_set = set()
            for _, row in gold_standard.iterrows():
                # Handle different column name formats
                if "tf" in row and "target" in row:
                    gs_set.add((row["tf"], row["target"]))
                elif "TF" in row and "Target" in row:
                    gs_set.add((row["TF"], row["Target"]))
                elif "source" in row and "target" in row:
                    gs_set.add((row["source"], row["target"]))
                elif len(row) >= 2:
                    # Assume first two columns are TF and target
                    gs_set.add((row.iloc[0], row.iloc[1]))
            gold_standard = gs_set

        # For DREAM5, filter to only genes in gold standard for efficiency
        gold_genes = set()
        for tf, tgt in gold_standard:
            gold_genes.add(tf)
            gold_genes.add(tgt)

        # Filter expression to gold standard genes only
        gold_gene_list = list(gold_genes & set(gene_names))
        gene_idx_map = {g: i for i, g in enumerate(gene_names)}
        gold_indices = [gene_idx_map[g] for g in gold_gene_list]

        expression_filtered = expression[:, gold_indices]
        gene_names_filtered = gold_gene_list

        # Update TF list to only include TFs in filtered genes
        tf_names_filtered = [tf for tf in tf_names if tf in gold_gene_list]

        preprocessor = GRNPreprocessor(normalization="zscore")
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression_filtered, gene_names_filtered, tf_names_filtered
        )
        all_target_genes = preprocessor.get_target_names()

        # Limit to first 30 targets for speed
        target_genes = all_target_genes[:30]

        # Filter expression to only include selected targets and all TFs
        target_indices_selected = [i for i, tgt in enumerate(all_target_genes) if tgt in target_genes]
        y_subset = y[:, target_indices_selected]

        # Also filter gold_standard to only include our subset of targets
        filtered_gold_standard = {
            (tf, tgt) for tf, tgt in gold_standard
            if tgt in target_genes and tf in tf_names_filtered
        }

        dataset_name = "DREAM5_Ecoli_subset"

        # Run TabPFN with all 5 strategies
        for strategy in tabpfn_strategies:
            result_tabpfn = run_tabpfnn_analysis_with_y(
                X, y_subset, tf_names_filtered, target_genes, filtered_gold_standard,
                dataset_name, n_estimators=1, edge_score_strategy=strategy
            )
            all_results.append(result_tabpfn)

        # Run correlation baseline with filtered expression
        # Need to create expression matrix with only TFs + selected targets
        tf_indices_filtered = [gene_names_filtered.index(tf) for tf in tf_names_filtered]
        target_indices_filtered = [gene_names_filtered.index(tgt) for tgt in target_genes]

        expression_for_baseline = expression_filtered[:, tf_indices_filtered + target_indices_filtered]
        gene_names_for_baseline = tf_names_filtered + target_genes

        result_corr = run_correlation_baseline(
            expression_for_baseline, gene_names_for_baseline, tf_names_filtered, filtered_gold_standard,
            target_genes, dataset_name
        )
        all_results.append(result_corr)

        # Run GENIE3 baseline
        result_genie3 = run_genie3_baseline(
            expression_for_baseline, gene_names_for_baseline, tf_names_filtered, filtered_gold_standard,
            target_genes, dataset_name
        )
        if result_genie3:
            all_results.append(result_genie3)

        # Run GRNBoost2 baseline
        result_grnboost2 = run_grnboost2_baseline(
            expression_for_baseline, gene_names_for_baseline, tf_names_filtered, filtered_gold_standard,
            target_genes, dataset_name
        )
        if result_grnboost2:
            all_results.append(result_grnboost2)
    except Exception as e:
        import traceback
        print(f"DREAM5 E. coli analysis skipped: {e}")
        traceback.print_exc()

    # ============================================================================
    # Generate Final Report
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)

    generate_comparison_report(all_results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
