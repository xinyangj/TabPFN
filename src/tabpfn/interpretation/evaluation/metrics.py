"""Evaluation metrics for the interpretation model."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)


def evaluate_binary(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate binary importance predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted importance scores in [0, 1], shape (n_features,) or (n_datasets, n_features).
    targets : np.ndarray
        Ground truth binary labels {0, 1}, same shape.
    threshold : float
        Decision threshold for binary predictions.

    Returns
    -------
    dict[str, float]
        Metrics: auroc, aupr, accuracy, precision, recall, f1.
    """
    pred_flat = predictions.ravel()
    target_flat = targets.ravel()

    # Remove padding (zeros in both)
    valid = ~((pred_flat == 0) & (target_flat == 0) & (predictions.ndim > 1))
    if predictions.ndim == 1:
        valid = np.ones_like(pred_flat, dtype=bool)

    pred_flat = pred_flat[valid]
    target_flat = target_flat[valid]

    metrics: dict[str, float] = {}

    # AUROC
    if len(np.unique(target_flat)) > 1:
        metrics["auroc"] = float(roc_auc_score(target_flat, pred_flat))
    else:
        metrics["auroc"] = 0.5

    # AUPR
    if len(np.unique(target_flat)) > 1:
        metrics["aupr"] = float(average_precision_score(target_flat, pred_flat))
    else:
        metrics["aupr"] = float(target_flat.mean())

    # Accuracy
    pred_binary = (pred_flat >= threshold).astype(float)
    metrics["accuracy"] = float(accuracy_score(target_flat, pred_binary))

    # Precision, Recall, F1
    tp = ((pred_binary == 1) & (target_flat == 1)).sum()
    fp = ((pred_binary == 1) & (target_flat == 0)).sum()
    fn = ((pred_binary == 0) & (target_flat == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    return metrics


def evaluate_continuous(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """Evaluate continuous importance predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted importance scores, shape (n_features,) or (n_datasets, n_features).
    targets : np.ndarray
        Ground truth importance scores, same shape.

    Returns
    -------
    dict[str, float]
        Metrics: r2, mae, correlation.
    """
    pred_flat = predictions.ravel()
    target_flat = targets.ravel()

    metrics: dict[str, float] = {}
    metrics["r2"] = float(r2_score(target_flat, pred_flat))
    metrics["mae"] = float(mean_absolute_error(target_flat, pred_flat))

    if np.std(pred_flat) > 0 and np.std(target_flat) > 0:
        metrics["correlation"] = float(np.corrcoef(pred_flat, target_flat)[0, 1])
    else:
        metrics["correlation"] = 0.0

    return metrics


def evaluate_ranking(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    top_k: int | None = None,
) -> dict[str, float]:
    """Evaluate ranking quality of importance predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted importance scores, shape (n_features,).
    targets : np.ndarray
        Ground truth importance, shape (n_features,).
    top_k : int, optional
        If specified, evaluate precision@k.

    Returns
    -------
    dict[str, float]
        Metrics: spearman_correlation, precision_at_k.
    """
    from scipy.stats import spearmanr

    metrics: dict[str, float] = {}

    # Spearman rank correlation
    if np.std(predictions) > 0 and np.std(targets) > 0:
        corr, _ = spearmanr(predictions, targets)
        metrics["spearman"] = float(corr)
    else:
        metrics["spearman"] = 0.0

    # Precision@k
    if top_k is not None and top_k > 0:
        pred_topk = set(np.argsort(predictions)[-top_k:])
        true_topk = set(np.argsort(targets)[-top_k:])
        if len(pred_topk) > 0:
            metrics[f"precision_at_{top_k}"] = len(pred_topk & true_topk) / len(
                pred_topk
            )
        else:
            metrics[f"precision_at_{top_k}"] = 0.0

    return metrics
