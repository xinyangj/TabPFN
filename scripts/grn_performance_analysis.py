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
import tempfile
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
from tabpfn.grn.baseline_models import TabPFNWrapper


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
# Wrapper Classes for Expression Prediction Evaluation
# ============================================================================

class GENIE3RegressorWrapper:
    """Wrapper for GENIE3 with fit/predict interface for expression evaluation.

    Uses Random Forest regression to predict each target gene from TFs.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models_: dict[int, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GENIE3RegressorWrapper":
        """Train one RandomForest per target gene.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)

        Returns
        -------
        self : GENIE3RegressorWrapper
        """
        from sklearn.ensemble import RandomForestRegressor

        n_targets = y.shape[1]
        for target_idx in range(n_targets):
            rf = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features="sqrt",
                random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X, y[:, target_idx])
            self.models_[target_idx] = rf

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression for all targets.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)
        """
        n_samples = X.shape[0]
        n_targets = len(self.models_)
        predictions = np.zeros((n_samples, n_targets))

        for target_idx, model in self.models_.items():
            predictions[:, target_idx] = model.predict(X)

        return predictions


class GRNBoost2RegressorWrapper:
    """Wrapper for GRNBoost2 with fit/predict interface for expression evaluation.

    Uses Gradient Boosting regression to predict each target gene from TFs.
    """

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models_: dict[int, Any] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GRNBoost2RegressorWrapper":
        """Train one GradientBoostingRegressor per target gene.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)

        Returns
        -------
        self : GRNBoost2RegressorWrapper
        """
        from sklearn.ensemble import GradientBoostingRegressor

        n_targets = y.shape[1]
        for target_idx in range(n_targets):
            gb = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=0.1,
                max_depth=3,
                random_state=self.random_state,
            )
            gb.fit(X, y[:, target_idx])
            self.models_[target_idx] = gb

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression for all targets.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)
        """
        n_samples = X.shape[0]
        n_targets = len(self.models_)
        predictions = np.zeros((n_samples, n_targets))

        for target_idx, model in self.models_.items():
            predictions[:, target_idx] = model.predict(X)

        return predictions


class CorrelationPredictorWrapper:
    """Linear regression wrapper for correlation-based expression prediction.

    Uses linear regression to predict target genes from TFs.
    The correlation method itself is used for edge scoring, but for
    expression prediction we need an actual predictive model.
    """

    def __init__(self):
        self.coef_: np.ndarray | None = None  # Shape: (n_TFs, n_targets)
        self.intercept_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationPredictorWrapper":
        """Fit linear regression: y = X @ coef_ + intercept_.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)

        Returns
        -------
        self : CorrelationPredictorWrapper
        """
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X, y)
        self.coef_ = lr.coef_.T  # Transpose to (n_TFs, n_targets)
        self.intercept_ = lr.intercept_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression: y_pred = X @ coef_ + intercept_.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before prediction")
        return X @ self.coef_ + self.intercept_


class MutualInfoPredictorWrapper:
    """Linear regression wrapper for MI-based expression prediction.

    Uses linear regression to predict target genes from TFs.
    The mutual information method is used for edge scoring, but for
    expression prediction we need an actual predictive model.
    """

    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MutualInfoPredictorWrapper":
        """Fit linear regression.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)

        Returns
        -------
        self : MutualInfoPredictorWrapper
        """
        from sklearn.linear_model import LinearRegression

        lr = LinearRegression()
        lr.fit(X, y)
        self.coef_ = lr.coef_.T
        self.intercept_ = lr.intercept_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict expression.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)
        """
        if self.coef_ is None or self.intercept_ is None:
            raise ValueError("Model must be fitted before prediction")
        return X @ self.coef_ + self.intercept_


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

    # Set the necessary attributes that predict_one_target_gene() needs
    model._tf_names = tf_names
    model._target_genes = target_genes

    # Initialize predictions array
    y_pred = np.zeros_like(y_test)

    # Fit, predict, and cleanup each target one at a time in a single loop
    for target_idx, target_name in enumerate(target_genes):
        y_target = y_train[:, target_idx]
        model.fit_one_target_gene(
            target_idx=target_idx,
            target_name=target_name,
            X_for_target=X_train,
            y_target=y_target,
            tf_names_for_target=tf_names,
        )

        # Predict for this target
        prediction_result = model.predict_one_target_gene(
            target_name=target_name,
            X=X_test,
            tf_names_for_target=tf_names,
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
    model_class: type,
    model_kwargs: dict,
    test_size: float = 0.2,
    random_state: int = 42,
    method_name: str = "Unknown",
) -> dict[str, Any]:
    """Evaluate expression prediction accuracy with train/test split.

    Evaluates how accurately a method predicts target gene expression values
    from input TF expression. Uses an 80/20 train/test split.

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
    model_class : type
        Model class with .fit() and .predict() methods
    model_kwargs : dict
        Keyword arguments for model initialization
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

    # Handle TabPFN - use TabPFNWrapper directly from baseline_models
    if "TabPFN" in str(model_class):
        n_estimators = model_kwargs.pop("n_estimators", 1)
        edge_score_strategy = model_kwargs.pop("edge_score_strategy", "self_attention")

        model = TabPFNWrapper(
            n_estimators=n_estimators,
            attention_aggregation="mean",
            edge_score_strategy=edge_score_strategy,
            device="auto",
            random_state=42,
            keep_model=True,  # Keep model for expression prediction, cleanup after each predict
        )
    else:
        model = model_class(**model_kwargs)

    # Train and evaluate
    start_time = time.time()

    # Special handling for TabPFNWrapper - use per-target methods
    if isinstance(model, TabPFNWrapper):
        metrics = evaluate_tabpfn_expression_prediction(
            model, X_train, y_train, X_test, y_test, target_genes, tf_names
        )
    else:
        metrics = evaluate_expression_prediction(
            model, X_train, y_train, X_test, y_test, target_genes
        )

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
                 "combined_best", "sequential_rollout", "gradient_rollout"],
        default=None,
        help="TabPFN edge score strategies to test (default: all 7 strategies)"
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=1,
        help="Number of TabPFN estimators (default: 1)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/grn_analysis)"
    )

    return parser.parse_args()


def main() -> None:
    """Run comprehensive GRN performance analysis."""
    args = parse_args()

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
    if args.output_dir:
        output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to run
    datasets_to_run = args.datasets
    if datasets_to_run is None:
        # Default: run all datasets
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
            "sequential_rollout",  # Unified attention rollout
            "gradient_rollout",    # Gradient-weighted attention rollout
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
    all_expression_results = []  # Collect all expression accuracy results

    # ============================================================================
    # DREAM4 Analysis (10 genes - fast for testing)
    # ============================================================================
    if "dream4-10" in datasets_to_run:
        print("\n" + "="*80)
        print("DREAM4 ANALYSIS (10 genes, 5 networks)")
        print("="*80)

        # Create loader for DREAM4 data
        dream4_loader = DREAMChallengeLoader(data_path=str(dream4_path))

        # Determine network IDs to run
        if args.network_ids:
            network_ids_to_run = args.network_ids
        elif args.max_networks:
            network_ids_to_run = range(1, min(args.max_networks + 1, 6))
        else:
            network_ids_to_run = range(1, 3)  # Default: 2 networks for speed

        for network_id in network_ids_to_run:
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

        # Run TabPFN with all strategies using unified runner
        # NEW: Use run_tabpfn_multiple_strategies() to fit ONCE and compute all strategies
        baseline_runner = GRNBaselineRunner(normalization="zscore")

        print(f"\n{'='*70}")
        print(f"TabPFN GRN Analysis: {dataset_name} (ALL STRATEGIES)")
        print(f"{'='*70}")
        print(f"  Running {len(tabpfn_strategies)} strategies with single fit...")

        # Fit once, compute all strategies
        tabpfn_results = baseline_runner.run_tabpfn_multiple_strategies(
            expression=expression,
            gene_names=gene_names,
            tf_names=tf_names,
            gold_standard=gold_standard,
            dataset_name=dataset_name,
            n_estimators=args.n_estimators,
            attention_aggregation="mean",
            edge_score_strategies=tabpfn_strategies,
        )

        # Add all results
        for strategy, result in tabpfn_results.items():
            all_results.append(result)
            strategy_label = strategy.replace("_", " ").title()
            print(f"  {strategy_label}: AUPR={result['metrics']['aupr']:.4f}, AUROC={result['metrics']['auroc']:.4f}")

        # Clean up GPU memory after all strategies
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

        # Expression Prediction Accuracy Evaluation (per network)
        if not args.no_expression:
            print(f"\n{'='*80}")
            print(f"EXPRESSION PREDICTION ACCURACY EVALUATION (DREAM4-10, network {network_id})")
            print(f"{'='*80}")

            # Note: All TabPFN strategies produce identical predictions (edge strategy only affects edge extraction)
            # So we only test one strategy for expression prediction
            result_tabpfn_expr = evaluate_expression_accuracy(
                X, y, tf_names, target_genes,
                model_class=TabPFNWrapper,
                model_kwargs={
                    "n_estimators": args.n_estimators,
                    "edge_score_strategy": "self_attention"
                },
                method_name="TabPFN"
            )
            result_tabpfn_expr["dataset"] = f"DREAM4_10_{network_id}_Expr"
            all_expression_results.append(result_tabpfn_expr)

            if not args.no_baselines:
                # GENIE3
                result_genie3_expr = evaluate_expression_accuracy(
                    X, y, tf_names, target_genes,
                    model_class=GENIE3RegressorWrapper,
                    model_kwargs={"n_estimators": 50},
                    method_name="GENIE3"
                )
                result_genie3_expr["dataset"] = f"DREAM4_10_{network_id}_Expr"
                all_expression_results.append(result_genie3_expr)

                # GRNBoost2
                result_grnboost2_expr = evaluate_expression_accuracy(
                    X, y, tf_names, target_genes,
                    model_class=GRNBoost2RegressorWrapper,
                    model_kwargs={"n_estimators": 50},
                    method_name="GRNBoost2"
                )
                result_grnboost2_expr["dataset"] = f"DREAM4_10_{network_id}_Expr"
                all_expression_results.append(result_grnboost2_expr)

                # Linear Regression (represents both Correlation and MI - both use LR for prediction)
                result_lr_expr = evaluate_expression_accuracy(
                    X, y, tf_names, target_genes,
                    model_class=CorrelationPredictorWrapper,
                    model_kwargs={},
                    method_name="Linear Regression"
                )
                result_lr_expr["dataset"] = f"DREAM4_10_{network_id}_Expr"
                all_expression_results.append(result_lr_expr)

    # ============================================================================
    # DREAM4 Analysis (100 genes - larger test)
    # ============================================================================
    if "dream4-100" in datasets_to_run:
        print("\n" + "="*80)
        print("DREAM4 ANALYSIS (100 genes)")
        print("="*80)

        # Create loader for DREAM4 data (if not already created)
        if 'dream4_loader' not in locals():
            dream4_loader = DREAMChallengeLoader(data_path=str(dream4_path))

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

        # Run TabPFN with all strategies using unified runner
        # NEW: Use run_tabpfn_multiple_strategies() to fit ONCE and compute all strategies
        baseline_runner = GRNBaselineRunner(normalization="zscore")

        print(f"\n{'='*70}")
        print(f"TabPFN GRN Analysis: {dataset_name} (ALL STRATEGIES)")
        print(f"{'='*70}")
        print(f"  Running {len(tabpfn_strategies)} strategies with single fit...")

        # Fit once, compute all strategies
        tabpfn_results = baseline_runner.run_tabpfn_multiple_strategies(
            expression=expression,
            gene_names=gene_names,
            tf_names=tf_names,
            gold_standard=gold_standard,
            dataset_name=dataset_name,
            n_estimators=args.n_estimators,
            attention_aggregation="mean",
            edge_score_strategies=tabpfn_strategies,
        )

        # Add all results
        for strategy, result in tabpfn_results.items():
            all_results.append(result)
            strategy_label = strategy.replace("_", " ").title()
            print(f"  {strategy_label}: AUPR={result['metrics']['aupr']:.4f}, AUROC={result['metrics']['auroc']:.4f}")

        # Clean up GPU memory after all strategies
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

        # Expression Prediction Accuracy Evaluation (DREAM4-100)
        if not args.no_expression:
            print(f"\n{'='*80}")
            print(f"EXPRESSION PREDICTION ACCURACY EVALUATION (DREAM4-100)")
            print(f"{'='*80}")

            # TabPFN (only one strategy needed - all produce identical predictions)
            result_tabpfn_expr = evaluate_expression_accuracy(
                X, y, tf_names, target_genes,
                model_class=TabPFNWrapper,
                model_kwargs={
                    "n_estimators": args.n_estimators,
                    "edge_score_strategy": "self_attention"
                },
                method_name="TabPFN"
            )
            result_tabpfn_expr["dataset"] = "DREAM4_100_Expr"
            all_expression_results.append(result_tabpfn_expr)

            if not args.no_baselines:
                # GENIE3
                result_genie3_expr = evaluate_expression_accuracy(
                    X, y, tf_names, target_genes,
                    model_class=GENIE3RegressorWrapper,
                    model_kwargs={"n_estimators": 50},
                    method_name="GENIE3"
                )
                result_genie3_expr["dataset"] = "DREAM4_100_Expr"
                all_expression_results.append(result_genie3_expr)

                # GRNBoost2
                result_grnboost2_expr = evaluate_expression_accuracy(
                    X, y, tf_names, target_genes,
                    model_class=GRNBoost2RegressorWrapper,
                    model_kwargs={"n_estimators": 50},
                    method_name="GRNBoost2"
                )
                result_grnboost2_expr["dataset"] = "DREAM4_100_Expr"
                all_expression_results.append(result_grnboost2_expr)

                # Linear Regression (represents both Correlation and MI)
                result_lr_expr = evaluate_expression_accuracy(
                    X, y, tf_names, target_genes,
                    model_class=CorrelationPredictorWrapper,
                    model_kwargs={},
                    method_name="Linear Regression"
                )
                result_lr_expr["dataset"] = "DREAM4_100_Expr"
                all_expression_results.append(result_lr_expr)

    # ============================================================================
    # DREAM5 E. coli Analysis (real data)
    # ============================================================================
    if "dream5" in datasets_to_run:
        print("\n" + "="*80)
        print("DREAM5 E. COLI ANALYSIS (real data)")
        print("="*80)

        # Clean up GPU memory before starting DREAM5 (largest dataset)
        print("Cleaning up GPU memory before DREAM5 analysis...")
        cleanup_gpu_memory()

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

            # Limit to user-specified number of targets for speed
            target_genes = all_target_genes[:args.max_targets]

            # Filter expression to only include selected targets and all TFs
            target_indices_selected = [i for i, tgt in enumerate(all_target_genes) if tgt in target_genes]
            y_subset = y[:, target_indices_selected]

            # Also filter gold_standard to only include our subset of targets
            filtered_gold_standard = {
                (tf, tgt) for tf, tgt in gold_standard
                if tgt in target_genes and tf in tf_names_filtered
            }

            dataset_name = "DREAM5_Ecoli_subset"

            # Need to create expression matrix with only TFs + selected targets
            # Do this BEFORE the TabPFN loop since these variables are used there
            tf_indices_filtered = [gene_names_filtered.index(tf) for tf in tf_names_filtered]
            target_indices_filtered = [gene_names_filtered.index(tgt) for tgt in target_genes]

            expression_for_baseline = expression_filtered[:, tf_indices_filtered + target_indices_filtered]
            gene_names_for_baseline = tf_names_filtered + target_genes

            # Run TabPFN with all strategies using unified runner
            # NEW: Use run_tabpfn_multiple_strategies() to fit ONCE and compute all strategies
            baseline_runner = GRNBaselineRunner(normalization="zscore")

            print(f"\n{'='*70}")
            print(f"TabPFN GRN Analysis: {dataset_name} (ALL STRATEGIES)")
            print(f"{'='*70}")
            print(f"  Running {len(tabpfn_strategies)} strategies with single fit...")

            # Fit once, compute all strategies
            tabpfn_results = baseline_runner.run_tabpfn_multiple_strategies(
                expression=expression_for_baseline,
                gene_names=gene_names_for_baseline,
                tf_names=tf_names_filtered,
                gold_standard=filtered_gold_standard,
                dataset_name=dataset_name,
                target_genes=target_genes,
                n_estimators=args.n_estimators,
                attention_aggregation="mean",
                edge_score_strategies=tabpfn_strategies,
            )

            # Add all results
            for strategy, result in tabpfn_results.items():
                all_results.append(result)
                strategy_label = strategy.replace("_", " ").title()
                print(f"  {strategy_label}: AUPR={result['metrics']['aupr']:.4f}, AUROC={result['metrics']['auroc']:.4f}")

            # Clean up GPU memory after all strategies (critical for DREAM5)
            cleanup_gpu_memory()

            # Run baseline methods using unified runner
            if not args.no_baselines:
                for method in ["correlation", "mutual_info", "genie3", "grnboost2"]:
                    result = baseline_runner.run_method(
                        method=method,
                        expression=expression_for_baseline,
                        gene_names=gene_names_for_baseline,
                        tf_names=tf_names_filtered,
                        gold_standard=filtered_gold_standard,
                        dataset_name=dataset_name,
                    )
                    if result is not None:
                        all_results.append(result)
                    cleanup_gpu_memory()
            cleanup_gpu_memory()

            # Expression Prediction Accuracy Evaluation (DREAM5)
            if not args.no_expression:
                print(f"\n{'='*80}")
                print(f"EXPRESSION PREDICTION ACCURACY EVALUATION (DREAM5 E. coli)")
                print(f"{'='*80}")

                # TabPFN (only one strategy needed)
                result_tabpfn_expr = evaluate_expression_accuracy(
                    X, y_subset, tf_names_filtered, target_genes,
                    model_class=TabPFNWrapper,
                    model_kwargs={
                        "n_estimators": args.n_estimators,
                        "edge_score_strategy": "self_attention"
                    },
                    method_name="TabPFN"
                )
                result_tabpfn_expr["dataset"] = "DREAM5_Ecoli_Expr"
                all_expression_results.append(result_tabpfn_expr)

                if not args.no_baselines:
                    # GENIE3
                    result_genie3_expr = evaluate_expression_accuracy(
                        X, y_subset, tf_names_filtered, target_genes,
                        model_class=GENIE3RegressorWrapper,
                        model_kwargs={"n_estimators": 50},
                        method_name="GENIE3"
                    )
                    result_genie3_expr["dataset"] = "DREAM5_Ecoli_Expr"
                    all_expression_results.append(result_genie3_expr)

                    # GRNBoost2
                    result_grnboost2_expr = evaluate_expression_accuracy(
                        X, y_subset, tf_names_filtered, target_genes,
                        model_class=GRNBoost2RegressorWrapper,
                        model_kwargs={"n_estimators": 50},
                        method_name="GRNBoost2"
                    )
                    result_grnboost2_expr["dataset"] = "DREAM5_Ecoli_Expr"
                    all_expression_results.append(result_grnboost2_expr)

                    # Linear Regression (represents both Correlation and MI)
                    result_lr_expr = evaluate_expression_accuracy(
                        X, y_subset, tf_names_filtered, target_genes,
                        model_class=CorrelationPredictorWrapper,
                        model_kwargs={},
                        method_name="Linear Regression"
                    )
                    result_lr_expr["dataset"] = "DREAM5_Ecoli_Expr"
                    all_expression_results.append(result_lr_expr)
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

    generate_comparison_report(all_results, output_dir, all_expression_results)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
