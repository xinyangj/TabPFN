"""Comprehensive GRN Performance Analysis Script.

Runs TabPFN GRN inference on DREAM datasets and compares against baseline methods.
Generates a detailed performance report with metrics and visualizations.

Usage:
    # Run all datasets
    python scripts/grn_performance_analysis.py

    # Run only specific datasets
    python scripts/grn_performance_analysis.py --datasets dream4-10 dream4-100

    # Run with limited networks for faster testing
    python scripts/grn_performance_analysis.py --datasets dream4-10 --max-networks 2

    # Skip expression evaluation
    python scripts/grn_performance_analysis.py --datasets dream4-10 --no-expression

    # Limit number of targets in DREAM5
    python scripts/grn_performance_analysis.py --datasets dream5 --max-targets 20
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    GRNBaselineRunner,
    GRNDataPipeline,
    TabPFNGRNRegressor,
    evaluate_grn,
    evaluate_expression_prediction,
    GRNNetworkVisualizer,
    EdgeScoreVisualizer,
    create_evaluation_summary_plot,
)
from tabpfn.grn.baseline_models import (
    SklearnForestWrapper,
    LinearRegressionWrapper,
    TabPFNWrapper,
)


def cleanup_gpu_memory() -> None:
    """Clean up GPU memory to prevent OOM errors during batch processing.

    This function:
    1. Forces Python garbage collection
    2. Clears PyTorch CUDA cache
    3. Resets peak memory stats
    """
    # Force Python garbage collection
    gc.collect()

    # Clear PyTorch CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Optional: Print memory stats for debugging
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"    GPU Memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


# ============================================================================
# GRN Analysis Functions (Edge Prediction) - Now using unified GRNBaselineRunner
# ============================================================================


def evaluate_tabpfn_expression_prediction(
    model: TabPFNWrapper,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_genes: list[str],
    tf_names: list[str],
) -> dict[str, float]:
    """Evaluate TabPFN expression prediction with memory-efficient per-target processing.

    Uses fit_one_target_gene() and predict_one_target_gene() in a single loop
    to fit/predict one target at a time with cleanup to avoid OOM.

    Parameters
    ----------
    model : TabPFNWrapper
        The TabPFN wrapper instance
    X_train : np.ndarray
        Training TF expression matrix (n_samples, n_TFs)
    y_train : np.ndarray
        Training target expression matrix (n_samples, n_targets)
    X_test : np.ndarray
        Test TF expression matrix (n_samples, n_TFs)
    y_test : np.ndarray
        Test target expression matrix (n_samples, n_targets)
    target_genes : list[str]
        Target gene names
    tf_names : list[str]
        Transcription factor names

    Returns
    -------
    metrics : dict
        Dictionary with mean/std of MSE, RMSE, MAE, R², Pearson r
    """
    import gc
    from tabpfn.grn.evaluation import compute_expression_metrics
    from tabpfn.grn.baseline_models import prepare_target_features

    # Set the necessary attributes that predict_one_target_gene() needs
    model._tf_names = tf_names
    model._target_genes = target_genes

    # Prepare features with proper target exclusion (prevents data leakage
    # when a target gene is also a TF, e.g., DREAM4-10 where all genes are TFs)
    prepared_train = prepare_target_features(
        X=X_train, y=y_train, tf_names=tf_names, target_genes=target_genes
    )

    # Initialize predictions array
    y_pred = np.zeros_like(y_test)

    # Fit, predict, and cleanup each target one at a time in a single loop
    for target_idx, target_name in enumerate(target_genes):
        X_train_for_target, tf_names_for_target = prepared_train[target_idx]
        y_target = y_train[:, target_idx]
        model.fit_one_target_gene(
            target_idx=target_idx,
            target_name=target_name,
            X_for_target=X_train_for_target,
            y_target=y_target,
            tf_names_for_target=tf_names_for_target,
        )

        # Predict for this target (using matching feature subset)
        prediction_result = model.predict_one_target_gene(
            target_name=target_name,
            X=X_test,
            tf_names_for_target=tf_names_for_target,
            cleanup_after=True,  # Cleanup after prediction
        )

        # predict_one_target_gene() returns a dict, extract the array
        if isinstance(prediction_result, dict):
            y_pred[:, target_idx] = prediction_result[target_name]
        else:
            y_pred[:, target_idx] = prediction_result

        # Force cleanup after each target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute metrics per target
    target_metrics = {}
    for i, target in enumerate(target_genes):
        target_metrics[target] = compute_expression_metrics(
            y_test[:, i], y_pred[:, i]
        )

    # Aggregate (mean and std across targets)
    aggregated = {}
    for metric in ["mse", "rmse", "mae", "r2", "pearson_r"]:
        values = [tm[metric] for tm in target_metrics.values()]
        aggregated[f"mean_{metric}"] = float(np.mean(values))
        aggregated[f"std_{metric}"] = float(np.std(values))

    # Final cleanup: Clear any remaining regressors from the wrapper
    # This ensures all GPU tensors are released
    if hasattr(model, '_regressors') and model._regressors:
        for target_name in list(model._regressors.keys()):
            if model._regressors[target_name] is not None:
                model._regressors[target_name].cleanup_model()
                del model._regressors[target_name]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return aggregated


def evaluate_expression_accuracy(
    X: np.ndarray,
    y: np.ndarray,
    tf_names: list[str],
    target_genes: list[str],
    model: Any,
    test_size: float = 0.2,
    random_state: int = 42,
    method_name: str = "Unknown",
) -> dict[str, Any]:
    """Evaluate expression prediction accuracy with train/test split.

    Evaluates how accurately a method predicts target gene expression values
    from input TF expression. Uses an 80/20 train/test split.

    All models are expected to come from baseline_models.py (SklearnForestWrapper,
    LinearRegressionWrapper, TabPFNWrapper) and share the same
    fit(X, y, tf_names, target_genes) / predict(X) API.

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
    model : Any
        Pre-constructed model instance with .fit() and .predict() methods.
        Should be one of SklearnForestWrapper, LinearRegressionWrapper, or TabPFNWrapper.
    test_size : float
        Fraction of data to use for testing (default 0.2)
    random_state : int
        Random seed for reproducibility
    method_name : str
        Name of the method for reporting

    Returns
    -------
    results : dict
        Dictionary with metrics and metadata:
        - 'method': Method name
        - 'metrics': Dict with mean/std of MSE, RMSE, MAE, R², Pearson r
        - 'evaluation_time': Time taken for evaluation
        - 'n_train': Number of training samples
        - 'n_test': Number of test samples
    """
    from sklearn.model_selection import train_test_split

    print(f"\n  Evaluating expression prediction: {method_name}")
    print(f"    Train/test split: {100*(1-test_size):.0f}/{100*test_size:.0f}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Test samples: {X_test.shape[0]}")

    # Train and evaluate
    start_time = time.time()

    # TabPFNWrapper needs memory-efficient per-target processing
    if isinstance(model, TabPFNWrapper):
        metrics = evaluate_tabpfn_expression_prediction(
            model, X_train, y_train, X_test, y_test, target_genes, tf_names
        )
    else:
        # SklearnForestWrapper and LinearRegressionWrapper accept tf_names/target_genes in fit()
        model.fit(X_train, y_train, tf_names=tf_names, target_genes=target_genes)
        y_pred = model.predict(X_test)

        # Compute metrics per target
        from tabpfn.grn.evaluation import compute_expression_metrics
        target_metrics = {}
        for i, target in enumerate(target_genes):
            target_metrics[target] = compute_expression_metrics(
                y_test[:, i], y_pred[:, i]
            )

        # Aggregate (mean and std across targets)
        metrics = {}
        for metric in ["mse", "rmse", "mae", "r2", "pearson_r"]:
            values = [tm[metric] for tm in target_metrics.values()]
            metrics[f"mean_{metric}"] = float(np.mean(values))
            metrics[f"std_{metric}"] = float(np.std(values))

    eval_time = time.time() - start_time

    # Add metadata
    metrics["evaluation_time"] = eval_time
    metrics["n_train"] = X_train.shape[0]
    metrics["n_test"] = X_test.shape[0]

    # Print results
    print(f"    Results:")
    print(f"      MSE:  {metrics['mean_mse']:.4f} (+/- {metrics['std_mse']:.4f})")
    print(f"      RMSE: {metrics['mean_rmse']:.4f} (+/- {metrics['std_rmse']:.4f})")
    print(f"      MAE:  {metrics['mean_mae']:.4f} (+/- {metrics['std_mae']:.4f})")
    print(f"      R²:   {metrics['mean_r2']:.4f} (+/- {metrics['std_r2']:.4f})")
    print(f"      Pearson r: {metrics['mean_pearson_r']:.4f} (+/- {metrics['std_pearson_r']:.4f})")

    return {
        "method": method_name,
        "metrics": metrics,
    }



# ============================================================================
# Report Generation
# ============================================================================

def generate_comparison_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    expression_results: list[dict[str, Any]] | None = None,
) -> None:
    """Generate comparison report and visualizations.

    Parameters
    ----------
    results : list of dict
        All edge prediction results from different methods
    output_dir : Path
        Directory to save outputs
    expression_results : list of dict, optional
        Expression prediction accuracy results
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by dataset
    datasets = {}
    for r in results:
        dataset = r["dataset"]
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(r)

    # Group expression results by dataset
    expression_datasets = {}
    if expression_results:
        for r in expression_results:
            dataset = r["dataset"]
            if dataset not in expression_datasets:
                expression_datasets[dataset] = []
            expression_datasets[dataset].append(r)

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

    # Add expression accuracy section to report
    if expression_datasets:
        report_lines.append("\n" + "=" * 80)
        report_lines.append("EXPRESSION PREDICTION ACCURACY")
        report_lines.append("=" * 80)
        report_lines.append("")

        for dataset_name, dataset_results in expression_datasets.items():
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"Dataset: {dataset_name}")
            report_lines.append(f"{'='*80}")
            report_lines.append("")

            # Create table header
            report_lines.append(f"{'Method':<25} {'R²':<10} {'Pearson r':<12} {'RMSE':<10}")
            report_lines.append("-" * 80)

            # Sort by R²
            sorted_results = sorted(
                dataset_results,
                key=lambda x: x["metrics"]["mean_r2"],
                reverse=True
            )

            for r in sorted_results:
                method = r["method"]
                r2 = r["metrics"]["mean_r2"]
                pearson = r["metrics"]["mean_pearson_r"]
                rmse = r["metrics"]["mean_rmse"]
                report_lines.append(f"{method:<25} {r2:<10.4f} {pearson:<12.4f} {rmse:<10.4f}")

            report_lines.append("")

        # Create expression accuracy comparison visualization
        if expression_datasets:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Collect all unique methods across datasets
            all_methods = sorted(set(
                r["method"] for dataset_results in expression_datasets.values()
                for r in dataset_results
            ))

            # Create color mapping for consistent colors across plots
            colors = plt.cm.tab10(np.linspace(0, 1, len(all_methods)))
            method_colors = {m: c for m, c in zip(all_methods, colors)}

            # Plot 1: R² comparison
            for dataset_name, dataset_results in expression_datasets.items():
                r2_values = {r["method"]: r["metrics"]["mean_r2"] for r in dataset_results}
                x_pos = list(expression_datasets.keys()).index(dataset_name)
                width = 0.8 / len(all_methods)
                for i, method in enumerate(all_methods):
                    if method in r2_values:
                        axes[0].bar(x_pos + i * width, r2_values[method], width,
                                  label=method if dataset_name == list(expression_datasets.keys())[0] else "",
                                  color=method_colors[method])

            axes[0].set_ylabel("R²")
            axes[0].set_title("Expression Prediction: R² Score")
            axes[0].set_xticks(np.arange(len(expression_datasets)))
            axes[0].set_xticklabels(expression_datasets.keys(), rotation=15, ha="right")
            axes[0].legend(fontsize=8, ncol=2)
            axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Pearson correlation
            for dataset_name, dataset_results in expression_datasets.items():
                pearson_values = {r["method"]: r["metrics"]["mean_pearson_r"] for r in dataset_results}
                x_pos = list(expression_datasets.keys()).index(dataset_name)
                for i, method in enumerate(all_methods):
                    if method in pearson_values:
                        axes[1].bar(x_pos + i * width, pearson_values[method], width,
                                  color=method_colors[method])

            axes[1].set_ylabel("Pearson r")
            axes[1].set_title("Expression Prediction: Pearson Correlation")
            axes[1].set_xticks(np.arange(len(expression_datasets)))
            axes[1].set_xticklabels(expression_datasets.keys(), rotation=15, ha="right")
            axes[1].axhline(y=0, color='k', linestyle='--', linewidth=0.5)
            axes[1].grid(True, alpha=0.3)

            # Plot 3: RMSE
            for dataset_name, dataset_results in expression_datasets.items():
                rmse_values = {r["method"]: r["metrics"]["mean_rmse"] for r in dataset_results}
                x_pos = list(expression_datasets.keys()).index(dataset_name)
                for i, method in enumerate(all_methods):
                    if method in rmse_values:
                        axes[2].bar(x_pos + i * width, rmse_values[method], width,
                                  color=method_colors[method])

            axes[2].set_ylabel("RMSE")
            axes[2].set_title("Expression Prediction: RMSE")
            axes[2].set_xticks(np.arange(len(expression_datasets)))
            axes[2].set_xticklabels(expression_datasets.keys(), rotation=15, ha="right")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "expression_accuracy_comparison.png", dpi=150)
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

    # Add expression accuracy to JSON summary
    if expression_datasets:
        summary["expression_accuracy"] = {}
        for dataset_name, dataset_results in expression_datasets.items():
            summary["expression_accuracy"][dataset_name] = {}
            for r in dataset_results:
                summary["expression_accuracy"][dataset_name][r["method"]] = {
                    "r2": r["metrics"]["mean_r2"],
                    "pearson_r": r["metrics"]["mean_pearson_r"],
                    "rmse": r["metrics"]["mean_rmse"],
                    "mae": r["metrics"]["mean_mae"],
                }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Report saved to: {report_path}")
    print(f"Summary saved to: {output_dir / 'summary.json'}")
    print(f"Expression accuracy plot saved to: {output_dir / 'expression_accuracy_comparison.png'}")
    print(f"{'='*80}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset selection.

    Returns
    -------
    args : argparse.Namespace
        Parsed arguments with dataset options
    """
    parser = argparse.ArgumentParser(
        description="GRN Performance Analysis - Compare TabPFN with baseline methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Run all datasets
  %(prog)s --datasets dream4-10 dream4-100    # Run only specific datasets
  %(prog)s --datasets dream4-10 --max-networks 2  # Limit networks for speed
  %(prog)s --datasets dream5 --max-targets 20     # Limit DREAM5 targets
  %(prog)s --no-expression --no-baselines        # Skip expression/baseline evaluation
        """
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dream4-10", "dream4-100", "dream5"],
        default=None,
        help="Datasets to run (default: all). Options: dream4-10, dream4-100, dream5"
    )

    parser.add_argument(
        "--max-networks",
        type=int,
        default=None,
        help="Maximum number of networks to process for DREAM4 (default: all 5 for 10-gene, 1 for 100-gene)"
    )

    parser.add_argument(
        "--network-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific network IDs to run for DREAM4 (1-indexed, e.g., --network-ids 1 3 5)"
    )

    parser.add_argument(
        "--max-targets",
        type=int,
        default=30,
        help="Maximum number of target genes for DREAM5 (default: 30)"
    )

    parser.add_argument(
        "--no-expression",
        action="store_true",
        help="Skip expression prediction accuracy evaluation"
    )

    parser.add_argument(
        "--no-baselines",
        action="store_true",
        help="Skip baseline methods (correlation, MI, GENIE3, GRNBoost2)"
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["self_attention", "tf_to_target", "target_to_tf", "combined",
                 "combined_best", "sequential_rollout", "gradient_rollout",
                 "integrated_gradients", "rise", "shapley"],
        default=None,
        help="TabPFN edge score strategies to test (default: all 9 strategies)"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1,
        help="Number of TabPFN estimators (default: 1)"
    )

    parser.add_argument(
        "--ig-n-folds",
        type=int,
        default=1,
        help="Number of CV folds for Integrated Gradients (default: 1, no CV)"
    )

    parser.add_argument(
        "--ig-baseline",
        type=str,
        default="zero",
        choices=["zero", "mean"],
        help="Baseline for Integrated Gradients (default: zero)"
    )

    parser.add_argument(
        "--rise-n-masks",
        type=int,
        default=500,
        help="Number of random masks for RISE (default: 500)"
    )

    parser.add_argument(
        "--rise-mask-prob",
        type=float,
        default=0.5,
        help="Probability of keeping each feature in RISE masks (default: 0.5)"
    )

    parser.add_argument(
        "--rise-baseline",
        type=str,
        default="zero",
        choices=["zero", "mean"],
        help="Baseline fill for masked features in RISE (default: zero)"
    )

    parser.add_argument(
        "--rise-n-folds",
        type=int,
        default=1,
        help="Number of CV folds for RISE (default: 1, no CV)"
    )

    parser.add_argument(
        "--shapley-n-permutations",
        type=int,
        default=200,
        help="Number of permutations for approximate Shapley (default: 200)"
    )

    parser.add_argument(
        "--shapley-n-folds",
        type=int,
        default=1,
        help="Number of CV folds for Shapley (default: 1, no CV)"
    )

    parser.add_argument(
        "--shapley-exact-threshold",
        type=int,
        default=15,
        help="Use exact Shapley when n_TFs <= this value (default: 15)"
    )

    parser.add_argument(
        "--shapley-method",
        type=str,
        default="auto",
        choices=["auto", "kernelshap_test", "kernelshap_train", "kernelshap_full"],
        help="Shapley computation method (default: auto = exact/permutation)"
    )

    parser.add_argument(
        "--shapley-n-coalitions",
        type=int,
        default=500,
        help="Max sampled coalitions for KernelSHAP (default: 500)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/grn_analysis)"
    )

    return parser.parse_args()


def load_dataset(
    dataset_key: str,
    args: argparse.Namespace,
    dream4_path: Path,
    dream5_path: Path,
) -> list[dict[str, Any]]:
    """Load and preprocess a DREAM dataset, returning standardized data dicts.

    Parameters
    ----------
    dataset_key : str
        One of 'dream4-10', 'dream4-100', 'dream5'
    args : argparse.Namespace
        Parsed command-line arguments
    dream4_path : Path
        Path to DREAM4 data directory
    dream5_path : Path
        Path to DREAM5 data directory

    Returns
    -------
    datasets : list of dict
        List of standardized dicts, one per network. Each dict has keys:
        expression, gene_names, tf_names, gold_standard, dataset_name,
        X, y, target_genes, expr_label
    """
    if dataset_key == "dream4-10":
        dream4_loader = DREAMChallengeLoader(data_path=str(dream4_path))

        # Determine network IDs to run
        if args.network_ids:
            network_ids_to_run = args.network_ids
        elif args.max_networks:
            network_ids_to_run = range(1, min(args.max_networks + 1, 6))
        else:
            network_ids_to_run = range(1, 6)  # Default: all 5 networks

        datasets = []
        for network_id in network_ids_to_run:
            expression, gene_names, tf_names, gold_standard = dream4_loader.load_dream4(
                network_size=10, network_id=network_id
            )

            preprocessor = GRNPreprocessor(normalization="zscore")
            X, y, tf_indices, target_indices = preprocessor.fit_transform(
                expression, gene_names, tf_names
            )
            target_genes = preprocessor.get_target_names()

            datasets.append({
                "expression": expression, "gene_names": gene_names,
                "tf_names": tf_names, "gold_standard": gold_standard,
                "dataset_name": f"DREAM4_10_{network_id}", "X": X, "y": y,
                "target_genes": target_genes,
                "expr_label": f"DREAM4_10_{network_id}_Expr",
                "target_genes_for_runner": None,
            })
        return datasets

    elif dataset_key == "dream4-100":
        dream4_loader = DREAMChallengeLoader(data_path=str(dream4_path))

        # Determine network IDs to run
        if args.network_ids:
            network_ids_to_run = args.network_ids
        elif args.max_networks:
            network_ids_to_run = range(1, min(args.max_networks + 1, 6))
        else:
            network_ids_to_run = [1]  # Default: network 1 (100-gene is expensive)

        datasets = []
        for network_id in network_ids_to_run:
            expression, gene_names, tf_names, gold_standard = dream4_loader.load_dream4(
                network_size=100, network_id=network_id
            )

            preprocessor = GRNPreprocessor(normalization="zscore")
            X, y, tf_indices, target_indices = preprocessor.fit_transform(
                expression, gene_names, tf_names
            )
            target_genes = preprocessor.get_target_names()

            datasets.append({
                "expression": expression, "gene_names": gene_names,
                "tf_names": tf_names, "gold_standard": gold_standard,
                "dataset_name": f"DREAM4_100_{network_id}", "X": X, "y": y,
                "target_genes": target_genes,
                "expr_label": f"DREAM4_100_{network_id}_Expr",
                "target_genes_for_runner": None,
            })
        return datasets

    elif dataset_key == "dream5":
        import pandas as pd

        # Clean up GPU memory before starting DREAM5 (largest dataset)
        print("Cleaning up GPU memory before DREAM5 analysis...")
        cleanup_gpu_memory()

        dream5_loader = DREAMChallengeLoader(data_path=str(dream5_path))
        expression, gene_names, tf_names, gold_standard = dream5_loader.load_dream5_ecoli()

        # Convert gold_standard DataFrame to set of positive edges only
        if isinstance(gold_standard, pd.DataFrame):
            # Filter to only positive edges (weight=1) if weight column exists
            if "weight" in gold_standard.columns:
                gold_standard = gold_standard[gold_standard["weight"] == 1]

            gs_set = set()
            for _, row in gold_standard.iterrows():
                if "tf" in row and "target" in row:
                    gs_set.add((row["tf"], row["target"]))
                elif "TF" in row and "Target" in row:
                    gs_set.add((row["TF"], row["Target"]))
                elif "source" in row and "target" in row:
                    gs_set.add((row["source"], row["target"]))
                elif len(row) >= 2:
                    gs_set.add((row.iloc[0], row.iloc[1]))
            gold_standard = gs_set

        # Filter to only genes in gold standard for efficiency
        gold_genes = set()
        for tf, tgt in gold_standard:
            gold_genes.add(tf)
            gold_genes.add(tgt)

        gold_gene_list = list(gold_genes & set(gene_names))
        gene_idx_map = {g: i for i, g in enumerate(gene_names)}
        gold_indices = [gene_idx_map[g] for g in gold_gene_list]

        expression_filtered = expression[:, gold_indices]
        gene_names_filtered = gold_gene_list
        tf_names_filtered = [tf for tf in tf_names if tf in gold_gene_list]

        preprocessor = GRNPreprocessor(normalization="zscore")
        X, y, tf_indices, target_indices = preprocessor.fit_transform(
            expression_filtered, gene_names_filtered, tf_names_filtered
        )
        all_target_genes = preprocessor.get_target_names()

        # Limit to user-specified number of targets for speed
        target_genes = all_target_genes[:args.max_targets]

        target_indices_selected = [i for i, tgt in enumerate(all_target_genes) if tgt in target_genes]
        y_subset = y[:, target_indices_selected]

        filtered_gold_standard = {
            (tf, tgt) for tf, tgt in gold_standard
            if tgt in target_genes and tf in tf_names_filtered
        }

        # Create expression matrix with only TFs + selected targets for baseline runner
        tf_indices_filtered = [gene_names_filtered.index(tf) for tf in tf_names_filtered]
        target_indices_filtered = [gene_names_filtered.index(tgt) for tgt in target_genes]
        expression_for_baseline = expression_filtered[:, tf_indices_filtered + target_indices_filtered]
        gene_names_for_baseline = tf_names_filtered + target_genes

        return [{
            "expression": expression_for_baseline,
            "gene_names": gene_names_for_baseline,
            "tf_names": tf_names_filtered,
            "gold_standard": filtered_gold_standard,
            "dataset_name": "DREAM5_Ecoli_subset",
            "X": X, "y": y_subset,
            "target_genes": target_genes,
            "expr_label": "DREAM5_Ecoli_Expr",
            "target_genes_for_runner": target_genes,
        }]
    else:
        raise ValueError(f"Unknown dataset: {dataset_key}")


def run_dataset_analysis(
    data: dict[str, Any],
    args: argparse.Namespace,
    tabpfn_strategies: list[str],
    all_results: list[dict],
    all_expression_results: list[dict],
) -> None:
    """Run the common analysis pipeline for a single dataset.

    Runs TabPFN strategies, baselines, and expression prediction evaluation,
    appending results to the provided lists.

    Parameters
    ----------
    data : dict
        Standardized data dict from load_dataset()
    args : argparse.Namespace
        Parsed command-line arguments
    tabpfn_strategies : list[str]
        TabPFN edge score strategies to test
    all_results : list[dict]
        Accumulator for edge prediction results (mutated in-place)
    all_expression_results : list[dict]
        Accumulator for expression prediction results (mutated in-place)
    """
    dataset_name = data["dataset_name"]
    expression = data["expression"]
    gene_names = data["gene_names"]
    tf_names = data["tf_names"]
    gold_standard = data["gold_standard"]
    X = data["X"]
    y = data["y"]
    target_genes = data["target_genes"]
    expr_label = data["expr_label"]
    target_genes_for_runner = data["target_genes_for_runner"]

    # Run TabPFN with all strategies using unified runner
    baseline_runner = GRNBaselineRunner(normalization="zscore")

    print(f"\n{'='*70}")
    print(f"TabPFN GRN Analysis: {dataset_name} (ALL STRATEGIES)")
    print(f"{'='*70}")
    print(f"  Running {len(tabpfn_strategies)} strategies with single fit...")

    tabpfn_results = baseline_runner.run_tabpfn_multiple_strategies(
        expression=expression,
        gene_names=gene_names,
        tf_names=tf_names,
        gold_standard=gold_standard,
        dataset_name=dataset_name,
        target_genes=target_genes_for_runner,
        n_estimators=args.n_estimators,
        attention_aggregation="mean",
        edge_score_strategies=tabpfn_strategies,
        ig_n_folds=args.ig_n_folds,
        ig_baseline=args.ig_baseline,
        rise_n_masks=args.rise_n_masks,
        rise_mask_prob=args.rise_mask_prob,
        rise_baseline=args.rise_baseline,
        rise_n_folds=args.rise_n_folds,
        shapley_n_permutations=args.shapley_n_permutations,
        shapley_n_folds=args.shapley_n_folds,
        shapley_exact_threshold=args.shapley_exact_threshold,
        shapley_method=args.shapley_method,
        shapley_n_coalitions=args.shapley_n_coalitions,
    )

    for strategy, result in tabpfn_results.items():
        all_results.append(result)
        strategy_label = strategy.replace("_", " ").title()
        print(f"  {strategy_label}: AUPR={result['metrics']['aupr']:.4f}, AUROC={result['metrics']['auroc']:.4f}")

    cleanup_gpu_memory()

    # Run baseline methods if enabled
    if not args.no_baselines:
        for method in ["correlation", "mutual_info", "genie3", "grnboost2"]:
            result = baseline_runner.run_method(
                method=method,
                expression=expression,
                gene_names=gene_names,
                tf_names=tf_names,
                gold_standard=gold_standard,
                dataset_name=dataset_name,
            )
            if result is not None:
                all_results.append(result)
            cleanup_gpu_memory()

    # Expression Prediction Accuracy Evaluation
    if not args.no_expression:
        print(f"\n{'='*80}")
        print(f"EXPRESSION PREDICTION ACCURACY EVALUATION ({dataset_name})")
        print(f"{'='*80}")

        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

        # TabPFN (only one strategy needed - all produce identical predictions)
        tabpfn_model = TabPFNWrapper(
            n_estimators=args.n_estimators,
            attention_aggregation="mean",
            edge_score_strategy="self_attention",
            device="auto",
            random_state=42,
            keep_model=True,
        )
        result_expr = evaluate_expression_accuracy(
            X, y, tf_names, target_genes,
            model=tabpfn_model,
            method_name="TabPFN"
        )
        result_expr["dataset"] = expr_label
        all_expression_results.append(result_expr)

        if not args.no_baselines:
            # GENIE3 (RandomForest)
            genie3_model = SklearnForestWrapper(
                estimator_class=RandomForestRegressor,
                estimator_kwargs={"max_features": "sqrt"},
                n_estimators=50,
                random_state=42,
            )
            result_expr = evaluate_expression_accuracy(
                X, y, tf_names, target_genes,
                model=genie3_model,
                method_name="GENIE3"
            )
            result_expr["dataset"] = expr_label
            all_expression_results.append(result_expr)

            # GRNBoost2 (GradientBoosting)
            grnboost2_model = SklearnForestWrapper(
                estimator_class=GradientBoostingRegressor,
                estimator_kwargs={"learning_rate": 0.1, "max_depth": 3},
                n_estimators=50,
                random_state=42,
            )
            result_expr = evaluate_expression_accuracy(
                X, y, tf_names, target_genes,
                model=grnboost2_model,
                method_name="GRNBoost2"
            )
            result_expr["dataset"] = expr_label
            all_expression_results.append(result_expr)

            # Linear Regression (represents both Correlation and MI)
            lr_model = LinearRegressionWrapper(random_state=42)
            result_expr = evaluate_expression_accuracy(
                X, y, tf_names, target_genes,
                model=lr_model,
                method_name="Linear Regression"
            )
            result_expr["dataset"] = expr_label
            all_expression_results.append(result_expr)


def main() -> None:
    """Run comprehensive GRN performance analysis."""
    args = parse_args()

    print("=" * 80)
    print("GRN PERFORMANCE ANALYSIS - TabPFN vs Baselines")
    print("=" * 80)

    # Setup - use local data directory
    project_root = Path(__file__).parent.parent
    dream4_path = project_root / "data" / "dream4"
    dream5_path = project_root / "data" / "dream5"
    output_dir = project_root / "results" / "grn_analysis"
    if args.output_dir:
        output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to run
    datasets_to_run = args.datasets
    if datasets_to_run is None:
        datasets_to_run = ["dream4-10", "dream4-100", "dream5"]

    # Define TabPFN edge score strategies to test
    tabpfn_strategies = args.strategies
    if tabpfn_strategies is None:
        tabpfn_strategies = [
            "self_attention",
            "tf_to_target",
            "target_to_tf",
            "combined",
            "combined_best",
            "sequential_rollout",
            "gradient_rollout",
            "integrated_gradients",
            "rise",
        ]

    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    print(f"  TabPFN strategies: {len(tabpfn_strategies)} strategies")
    print(f"  Expression evaluation: {'Disabled' if args.no_expression else 'Enabled'}")
    print(f"  Baseline methods: {'Disabled' if args.no_baselines else 'Enabled'}")
    print(f"  Estimators: {args.n_estimators}")
    print(f"  Output directory: {output_dir}")

    all_results = []
    all_expression_results = []

    # Run each dataset through the unified pipeline
    for dataset_key in datasets_to_run:
        print("\n" + "="*80)
        print(f"DATASET: {dataset_key.upper()}")
        print("="*80)

        try:
            datasets = load_dataset(dataset_key, args, dream4_path, dream5_path)
            for data in datasets:
                print(f"\n  >> Running: {data['dataset_name']}")
                run_dataset_analysis(
                    data, args, tabpfn_strategies, all_results, all_expression_results
                )
        except Exception as e:
            import traceback
            print(f"{dataset_key} analysis skipped: {e}")
            traceback.print_exc()

    # ============================================================================
    # Generate Final Report
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING FINAL REPORT")
    print("="*80)

    generate_comparison_report(all_results, output_dir, all_expression_results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
