"""Comprehensive GRN Performance Analysis Script.

Runs TabPFN GRN inference on DREAM datasets and compares against baseline methods.
Generates a detailed performance report with metrics and visualizations.
"""

from __future__ import annotations

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
    TabPFNGRNRegressor,
    evaluate_grn,
    evaluate_expression_prediction,
    GRNNetworkVisualizer,
    EdgeScoreVisualizer,
    create_evaluation_summary_plot,
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
# Wrapper Classes for Expression Prediction Evaluation
# ============================================================================

class TabPFNRegressorWrapper:
    """Wrapper for TabPFNGRNRegressor to provide consistent predict() interface.

    This wrapper converts TabPFN's dict output format to numpy array format
    for compatibility with the expression evaluation framework.
    """

    def __init__(
        self,
        tf_names: list[str],
        target_genes: list[str],
        n_estimators: int = 1,
        edge_score_strategy: str = "self_attention",
    ):
        from tabpfn.grn import TabPFNGRNRegressor

        self.tf_names = tf_names
        self.target_genes = target_genes
        self.model = TabPFNGRNRegressor(
            tf_names=tf_names,
            target_genes=target_genes,
            n_estimators=n_estimators,
            edge_score_strategy=edge_score_strategy,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TabPFNRegressorWrapper":
        """Fit the TabPFN GRN model.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)
        y : np.ndarray
            Target expression matrix (n_samples, n_targets)

        Returns
        -------
        self : TabPFNRegressorWrapper
        """
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target gene expression.

        Parameters
        ----------
        X : np.ndarray
            TF expression matrix (n_samples, n_TFs)

        Returns
        -------
        predictions : np.ndarray
            Predicted expression of shape (n_samples, n_targets)
        """
        pred_dict = self.model.predict(X)
        # Convert dict to array: column_stack preserves order of target_genes
        return np.column_stack([pred_dict[t] for t in self.target_genes])


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
# GRN Analysis Functions (Edge Prediction)
# ============================================================================


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
        device="auto",  # Automatically use CUDA if available
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

    # CRITICAL: Release model memory before returning
    # Delete model and clear attention weights to free GPU memory
    del grn_model
    cleanup_gpu_memory()

    return {
        "dataset": dataset_name,
        "method": f"TabPFN ({strategy_label})",
        "metrics": metrics,
        "edge_scores": edge_scores,
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
        device="auto",  # Automatically use CUDA if available
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

    # CRITICAL: Release model memory before returning
    # Delete model and clear attention weights to free GPU memory
    del grn_model
    cleanup_gpu_memory()

    return {
        "dataset": dataset_name,
        "method": f"TabPFN ({strategy_label})",
        "metrics": metrics,
        "edge_scores": edge_scores,
        "strategy": edge_score_strategy,
    }


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

    # Handle TabPFN wrapper which needs tf_names and target_genes
    if "TabPFNRegressorWrapper" in str(model_class):
        # Extract TabPFN-specific kwargs
        n_estimators = model_kwargs.pop("n_estimators", 1)
        edge_score_strategy = model_kwargs.pop("edge_score_strategy", "self_attention")
        model = model_class(
            tf_names=tf_names,
            target_genes=target_genes,
            n_estimators=n_estimators,
            edge_score_strategy=edge_score_strategy,
        )
    else:
        model = model_class(**model_kwargs)

    # Train and evaluate
    start_time = time.time()
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
# GRN Analysis Functions (Edge Prediction)
# ============================================================================

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
    all_expression_results = []  # Collect all expression accuracy results

    # Define TabPFN edge score strategies to test
    tabpfn_strategies = [
        "self_attention",
        "tf_to_target",
        "target_to_tf",
        "combined",
        "combined_best",
        "sequential_rollout",  # Unified attention rollout
        "gradient_rollout",    # NEW: Gradient-weighted attention rollout (GMAR-inspired)
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
            # Clean up GPU memory after each strategy
            cleanup_gpu_memory()

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

        # Expression Prediction Accuracy Evaluation (per network)
        print(f"\n{'='*80}")
        print(f"EXPRESSION PREDICTION ACCURACY EVALUATION (DREAM4-10, network {network_id})")
        print(f"{'='*80}")

        # Note: All TabPFN strategies produce identical predictions (edge strategy only affects edge extraction)
        # So we only test one strategy for expression prediction
        result_tabpfn_expr = evaluate_expression_accuracy(
            X, y, tf_names, target_genes,
            model_class=TabPFNRegressorWrapper,
            model_kwargs={
                "n_estimators": 1,
                "edge_score_strategy": "self_attention"
            },
            method_name="TabPFN"
        )
        result_tabpfn_expr["dataset"] = f"DREAM4_10_{network_id}_Expr"
        all_expression_results.append(result_tabpfn_expr)

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
        # Clean up GPU memory after each strategy
        cleanup_gpu_memory()

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

    # Run Mutual Information baseline
    result_mi = run_mutual_information_baseline(
        expression, gene_names, tf_names, gold_standard,
        target_genes, dataset_name
    )
    all_results.append(result_mi)

    # Expression Prediction Accuracy Evaluation (DREAM4-100)
    print(f"\n{'='*80}")
    print(f"EXPRESSION PREDICTION ACCURACY EVALUATION (DREAM4-100)")
    print(f"{'='*80}")

    # TabPFN (only one strategy needed - all produce identical predictions)
    result_tabpfn_expr = evaluate_expression_accuracy(
        X, y, tf_names, target_genes,
        model_class=TabPFNRegressorWrapper,
        model_kwargs={
            "n_estimators": 1,
            "edge_score_strategy": "self_attention"
        },
        method_name="TabPFN"
    )
    result_tabpfn_expr["dataset"] = "DREAM4_100_Expr"
    all_expression_results.append(result_tabpfn_expr)

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

        # Run TabPFN with all 7 strategies
        for strategy in tabpfn_strategies:
            result_tabpfn = run_tabpfnn_analysis_with_y(
                X, y_subset, tf_names_filtered, target_genes, filtered_gold_standard,
                dataset_name, n_estimators=1, edge_score_strategy=strategy
            )
            all_results.append(result_tabpfn)
            # Clean up GPU memory after each strategy (critical for DREAM5)
            cleanup_gpu_memory()

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
        cleanup_gpu_memory()

        # Run GENIE3 baseline
        result_genie3 = run_genie3_baseline(
            expression_for_baseline, gene_names_for_baseline, tf_names_filtered, filtered_gold_standard,
            target_genes, dataset_name
        )
        if result_genie3:
            all_results.append(result_genie3)
        cleanup_gpu_memory()

        # Run GRNBoost2 baseline
        result_grnboost2 = run_grnboost2_baseline(
            expression_for_baseline, gene_names_for_baseline, tf_names_filtered, filtered_gold_standard,
            target_genes, dataset_name
        )
        if result_grnboost2:
            all_results.append(result_grnboost2)
        cleanup_gpu_memory()

        # Run Mutual Information baseline
        result_mi = run_mutual_information_baseline(
            expression_for_baseline, gene_names_for_baseline, tf_names_filtered, filtered_gold_standard,
            target_genes, dataset_name
        )
        all_results.append(result_mi)
        cleanup_gpu_memory()

        # Expression Prediction Accuracy Evaluation (DREAM5)
        print(f"\n{'='*80}")
        print(f"EXPRESSION PREDICTION ACCURACY EVALUATION (DREAM5 E. coli)")
        print(f"{'='*80}")

        # TabPFN (only one strategy needed)
        result_tabpfn_expr = evaluate_expression_accuracy(
            X, y_subset, tf_names_filtered, target_genes,
            model_class=TabPFNRegressorWrapper,
            model_kwargs={
                "n_estimators": 1,
                "edge_score_strategy": "self_attention"
            },
            method_name="TabPFN"
        )
        result_tabpfn_expr["dataset"] = "DREAM5_Ecoli_Expr"
        all_expression_results.append(result_tabpfn_expr)

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
