#!/usr/bin/env python
"""Evaluate RF-based baselines (GENIE3/GRNBoost2) vs interpretation model on synthetic test data.

For each synthetic test dataset:
  - Train RandomForestRegressor (GENIE3-style) on raw_X_train → raw_y_train
  - Train GradientBoostingRegressor (GRNBoost2-style) on raw_X_train → raw_y_train
  - Extract feature_importances_ as predicted importance scores
  - Also run the trained interpretation model on the same data
  - Compare all against label_binary_direct (ground truth from DAG) using AUROC, AUPR

Also evaluates on DREAM4 benchmarks if cached features are available.

Usage:
    python scripts/eval_rf_baseline.py [--device cuda:0] [--max_datasets 0] [--n_estimators 100]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/interpretation_cache_v2")
MODEL_PATH = Path("results/interpretation_experiments/best_v2_model.pt")
DREAM_DIR = Path("data/dream_interpretation_features")
OUT_DIR = Path("results/interpretation_experiments/rf_baseline")

INPUT_CATEGORIES = sorted([
    "between_features_attention",
    "between_items_attention",
    "mlp_activations",
    "gradients",
])
EXPECTED_DIM = 691


# ─── Data loading (matching retrain_best_v2.py split) ───


def load_and_split_data(cache_dir: Path):
    """Load v2 cache and split into train/val/test (70/15/15, sequential)."""
    from scripts.generate_interpretation_data import load_all_datasets
    all_data = load_all_datasets(cache_dir)
    n = len(all_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    test_data = all_data[n_train + n_val:]
    logger.info(f"Total: {n}, Test split: {len(test_data)} datasets")
    return test_data


def get_features_and_labels(record: dict):
    """Extract 691-dim features and binary labels from a dataset record."""
    cat_vecs = record.get("category_vectors", {})
    parts = [cat_vecs[c] for c in INPUT_CATEGORIES if c in cat_vecs]
    if not parts:
        return None, None
    fv = np.concatenate(parts, axis=1)
    if fv.shape[1] != EXPECTED_DIM:
        return None, None
    labels = record["labels"].get("binary_direct")
    if labels is None:
        return None, None
    return fv, labels


def get_raw_data(record: dict):
    """Extract raw X_train, y_train from a dataset record."""
    raw = record.get("raw_data", {})
    X_train = raw.get("X_train")
    y_train = raw.get("y_train")
    if X_train is None or y_train is None:
        return None, None
    return X_train, y_train


# ─── Interpretation model ───


def load_interpretation_model(model_path: Path, device: str):
    """Load trained interpretation model."""
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = InterpretationModel(
        variant=checkpoint["variant"],
        input_dim=checkpoint["input_dim"],
        output_mode=checkpoint["output_mode"],
        hidden_dims=checkpoint["hidden_dims"],
        dropout=checkpoint["dropout"],
        norm=checkpoint["norm"],
        activation=checkpoint["activation"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    # Note: do NOT apply global_norm_stats — model was trained without them
    model.eval()
    return model


def predict_interpretation(model, feature_vectors: np.ndarray, device: str) -> np.ndarray:
    """Run interpretation model on feature vectors → importance scores."""
    fv_tensor = torch.tensor(feature_vectors, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(fv_tensor)
    scores = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    return scores


# ─── RF/GB baselines ───


def train_rf_and_get_importances(X_train: np.ndarray, y_train: np.ndarray,
                                  n_estimators: int = 100) -> np.ndarray:
    """Train RandomForestRegressor (GENIE3-style) and return feature importances."""
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    return rf.feature_importances_


def train_gb_and_get_importances(X_train: np.ndarray, y_train: np.ndarray,
                                  n_estimators: int = 100) -> np.ndarray:
    """Train GradientBoostingRegressor (GRNBoost2-style) and return feature importances."""
    gb = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    return gb.feature_importances_


# ─── Metrics ───


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute AUROC and AUPR. Returns NaN if degenerate (all same label)."""
    if len(np.unique(y_true)) < 2:
        return {"auroc": float("nan"), "aupr": float("nan")}
    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)
    return {"auroc": auroc, "aupr": aupr}


# ─── Synthetic data evaluation ───


def evaluate_on_synthetic(test_data: list, model, device: str,
                          n_estimators: int = 100, max_datasets: int = 0):
    """Evaluate all methods on synthetic test datasets."""
    results = {
        "interpretation_model": [],
        "random_forest": [],
        "gradient_boosting": [],
        "random": [],
    }
    # Collect all predictions for flattened (pooled) AUROC computation
    flat_preds = {m: [] for m in results}
    flat_labels = {m: [] for m in results}
    metadata_list = []

    n_total = len(test_data) if max_datasets <= 0 else min(max_datasets, len(test_data))
    n_skipped = 0
    t0 = time.time()

    for i in range(n_total):
        record = test_data[i]
        fv, labels = get_features_and_labels(record)
        X_train, y_train = get_raw_data(record)

        if fv is None or X_train is None:
            n_skipped += 1
            continue

        n_features = fv.shape[0]
        if n_features < 3:
            n_skipped += 1
            continue

        # Interpretation model
        interp_scores = predict_interpretation(model, fv, device)

        # RF (GENIE3)
        rf_importances = train_rf_and_get_importances(X_train, y_train, n_estimators)

        # GB (GRNBoost2)
        gb_importances = train_gb_and_get_importances(X_train, y_train, n_estimators)

        # Random baseline
        rng = np.random.RandomState(i)
        random_scores = rng.random(n_features)

        # Compute per-dataset metrics
        results["interpretation_model"].append(compute_metrics(labels, interp_scores))
        results["random_forest"].append(compute_metrics(labels, rf_importances))
        results["gradient_boosting"].append(compute_metrics(labels, gb_importances))
        results["random"].append(compute_metrics(labels, random_scores))

        # Accumulate for flattened metrics
        for method, scores in [("interpretation_model", interp_scores),
                               ("random_forest", rf_importances),
                               ("gradient_boosting", gb_importances),
                               ("random", random_scores)]:
            flat_preds[method].extend(scores.tolist())
            flat_labels[method].extend(labels.tolist())

        metadata_list.append({
            "n_features": int(n_features),
            "n_train_samples": int(X_train.shape[0]),
            "n_positive": int(labels.sum()),
            "edge_density": float(labels.mean()),
        })

        evaluated = len(results["interpretation_model"])
        if evaluated % 1000 == 0:
            elapsed = time.time() - t0
            rate = evaluated / elapsed
            logger.info(
                f"  {evaluated}/{n_total} evaluated ({elapsed:.0f}s, {rate:.1f}/s)"
            )

    elapsed = time.time() - t0
    evaluated = len(results["interpretation_model"])
    logger.info(
        f"Evaluated {evaluated} datasets, skipped {n_skipped} "
        f"({elapsed:.1f}s, {evaluated/max(elapsed,1):.1f}/s)"
    )

    return results, metadata_list, flat_preds, flat_labels


def aggregate_results(results: dict, metadata_list: list,
                      flat_preds: dict = None, flat_labels: dict = None) -> dict:
    """Aggregate per-dataset metrics into summary statistics."""
    summary = {}

    for method, metrics_list in results.items():
        aurocs = [m["auroc"] for m in metrics_list if not np.isnan(m["auroc"])]
        auprs = [m["aupr"] for m in metrics_list if not np.isnan(m["aupr"])]
        entry = {
            "auroc_mean": float(np.mean(aurocs)) if aurocs else 0,
            "auroc_std": float(np.std(aurocs)) if aurocs else 0,
            "auroc_median": float(np.median(aurocs)) if aurocs else 0,
            "aupr_mean": float(np.mean(auprs)) if auprs else 0,
            "aupr_std": float(np.std(auprs)) if auprs else 0,
            "n_valid": len(aurocs),
        }
        # Flattened (pooled) metrics — comparable to retrain-style evaluation
        if flat_preds and flat_labels and method in flat_preds:
            fp = np.array(flat_preds[method])
            fl = np.array(flat_labels[method])
            if len(np.unique(fl)) >= 2:
                entry["auroc_pooled"] = float(roc_auc_score(fl, fp))
                entry["aupr_pooled"] = float(average_precision_score(fl, fp))
                entry["n_features_total"] = int(len(fp))
        summary[method] = entry

    # Breakdown by feature count buckets
    buckets = {"3-10": [], "10-20": [], "20-30": [], "30+": []}
    for i, meta in enumerate(metadata_list):
        nf = meta["n_features"]
        if nf <= 10:
            buckets["3-10"].append(i)
        elif nf <= 20:
            buckets["10-20"].append(i)
        elif nf <= 30:
            buckets["20-30"].append(i)
        else:
            buckets["30+"].append(i)

    bucket_summary = {}
    for bname, indices in buckets.items():
        if not indices:
            bucket_summary[bname] = {"n": 0}
            continue
        bucket_summary[bname] = {"n": len(indices)}
        for method, metrics_list in results.items():
            aurocs = [metrics_list[i]["auroc"] for i in indices
                      if not np.isnan(metrics_list[i]["auroc"])]
            bucket_summary[bname][method] = {
                "auroc_mean": float(np.mean(aurocs)) if aurocs else 0,
                "auroc_std": float(np.std(aurocs)) if aurocs else 0,
            }

    return {"overall": summary, "by_feature_count": bucket_summary}


# ─── DREAM4 evaluation ───


def load_dream4_expression(network_size: int, network_id: int):
    """Load DREAM4 expression matrix using DREAMChallengeLoader."""
    from tabpfn.grn.datasets import DREAMChallengeLoader
    loader = DREAMChallengeLoader(data_path="data/dream4")
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=network_size, network_id=network_id
    )
    return expression, gene_names, tf_names, gold_standard


def evaluate_on_dream4(model, device: str, n_estimators: int = 100):
    """Evaluate RF/GB baselines on DREAM4 data alongside the interpretation model."""
    if not DREAM_DIR.exists():
        logger.warning(f"DREAM features dir not found: {DREAM_DIR}")
        return {}

    dream_results = {}

    for size in [10, 100]:
        size_results = []
        for net_id in range(1, 6):
            npz_path = DREAM_DIR / f"dream4_{size}_net{net_id}.npz"
            if not npz_path.exists():
                logger.warning(f"Missing {npz_path}")
                continue

            data = np.load(npz_path, allow_pickle=True)
            gene_names = list(data["gene_names"])
            tf_names = list(data["tf_names"])
            target_genes = list(data["target_genes"])
            gold_tf = list(data["gold_tf"])
            gold_target = list(data["gold_target"])
            gold_weight = data["gold_weight"]
            gold_edges = set()
            for i in range(len(gold_tf)):
                if gold_weight[i] > 0:
                    gold_edges.add((gold_tf[i], gold_target[i]))

            # Load expression matrix for RF/GB
            try:
                expression, _, _, _ = load_dream4_expression(size, net_id)
            except Exception as e:
                logger.warning(f"Could not load DREAM4-{size} net{net_id} expression: {e}")
                expression = None

            gene_to_idx = {g: i for i, g in enumerate(gene_names)}

            net_scores = {"network": f"dream4_{size}_net{net_id}"}
            method_edge_scores = {}

            for target in target_genes:
                target_key = f"fv_{target}"
                tfs_key = f"tfs_{target}"
                if target_key not in data.files or tfs_key not in data.files:
                    continue

                fv = data[target_key]  # (n_tfs, D)
                tfs = list(data[tfs_key])

                if fv.shape[1] != EXPECTED_DIM:
                    if fv.shape[1] > EXPECTED_DIM:
                        fv = fv[:, :EXPECTED_DIM]
                    else:
                        continue

                # Interpretation model scores
                interp_scores = predict_interpretation(model, fv, device)
                for j, tf in enumerate(tfs):
                    if tf != target:
                        method_edge_scores.setdefault("interpretation_model", {})[
                            (tf, target)
                        ] = float(interp_scores[j])

                # RF/GB on expression data
                if expression is not None and target in gene_to_idx:
                    target_idx = gene_to_idx[target]
                    y = expression[:, target_idx]

                    tf_indices = []
                    tf_names_for_target = []
                    for tf in tfs:
                        if tf != target and tf in gene_to_idx:
                            tf_indices.append(gene_to_idx[tf])
                            tf_names_for_target.append(tf)

                    if len(tf_indices) >= 2:
                        X = expression[:, tf_indices]

                        rf_imp = train_rf_and_get_importances(X, y, n_estimators)
                        for j, tf in enumerate(tf_names_for_target):
                            method_edge_scores.setdefault("random_forest", {})[
                                (tf, target)
                            ] = float(rf_imp[j])

                        gb_imp = train_gb_and_get_importances(X, y, n_estimators)
                        for j, tf in enumerate(tf_names_for_target):
                            method_edge_scores.setdefault("gradient_boosting", {})[
                                (tf, target)
                            ] = float(gb_imp[j])

            # Evaluate each method
            all_possible = set()
            for tf in tf_names:
                for tg in target_genes:
                    if tf != tg:
                        all_possible.add((tf, tg))

            net_results = {}
            y_true = np.array([1 if e in gold_edges else 0 for e in sorted(all_possible)])

            for method, edge_scores in method_edge_scores.items():
                y_score = np.array([edge_scores.get(e, 0.0) for e in sorted(all_possible)])
                if len(np.unique(y_true)) >= 2:
                    net_results[method] = {
                        "auroc": float(roc_auc_score(y_true, y_score)),
                        "aupr": float(average_precision_score(y_true, y_score)),
                    }

            # Random baseline
            rng = np.random.RandomState(net_id)
            y_score_rand = rng.random(len(all_possible))
            net_results["random"] = {
                "auroc": float(roc_auc_score(y_true, y_score_rand)),
                "aupr": float(average_precision_score(y_true, y_score_rand)),
            }

            # Attention heuristic (from cached features)
            attn_scores = {}
            for target in target_genes:
                tfs_key = f"tfs_{target}"
                fv_key = f"fv_{target}"
                if tfs_key in data.files and fv_key in data.files:
                    tfs_list = list(data[tfs_key])
                    fv_attn = data[fv_key]
                    attn_part = fv_attn[:, :327]
                    attn_score = np.mean(np.abs(attn_part), axis=1)
                    for j, tf in enumerate(tfs_list):
                        if tf != target:
                            attn_scores[(tf, target)] = float(attn_score[j])
            if attn_scores:
                y_score_attn = np.array([attn_scores.get(e, 0.0) for e in sorted(all_possible)])
                net_results["attention_heuristic"] = {
                    "auroc": float(roc_auc_score(y_true, y_score_attn)),
                    "aupr": float(average_precision_score(y_true, y_score_attn)),
                }

            net_scores["results"] = net_results
            size_results.append(net_scores)
            logger.info(
                f"  DREAM4-{size} net{net_id}: "
                + ", ".join(f"{m}={r['auroc']:.4f}" for m, r in net_results.items())
            )

        # Aggregate across 5 networks
        if size_results:
            agg = {}
            all_methods = set()
            for nr in size_results:
                all_methods.update(nr["results"].keys())
            for method in sorted(all_methods):
                aurocs = [nr["results"][method]["auroc"]
                          for nr in size_results if method in nr["results"]]
                auprs = [nr["results"][method]["aupr"]
                         for nr in size_results if method in nr["results"]]
                agg[method] = {
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "aupr_mean": float(np.mean(auprs)),
                    "aupr_std": float(np.std(auprs)),
                    "n_networks": len(aurocs),
                }
            dream_results[f"dream4_{size}"] = {
                "per_network": size_results,
                "summary": agg,
            }

    return dream_results


# ─── Main ───


def main():
    parser = argparse.ArgumentParser(description="RF baseline comparison")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_datasets", type=int, default=0,
                        help="Max test datasets to evaluate (0=all)")
    parser.add_argument("--n_estimators", type=int, default=100,
                        help="Number of trees for RF/GB")
    parser.add_argument("--skip_synthetic", action="store_true",
                        help="Skip synthetic data evaluation")
    parser.add_argument("--skip_dream", action="store_true",
                        help="Skip DREAM4 evaluation")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to interpretation model checkpoint")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model_path) if args.model_path else MODEL_PATH
    logger.info(f"Device: {device}, Model: {model_path}")

    # Load interpretation model
    logger.info("Loading interpretation model...")
    model = load_interpretation_model(model_path, device)

    all_results = {}

    # ─── Synthetic test data evaluation ───
    if not args.skip_synthetic:
        logger.info(f"Loading v2 cache from {CACHE_DIR}...")
        test_data = load_and_split_data(CACHE_DIR)

        logger.info(f"Evaluating on {len(test_data)} synthetic test datasets...")
        raw_results, metadata_list, flat_preds, flat_labels = evaluate_on_synthetic(
            test_data, model, device,
            n_estimators=args.n_estimators,
            max_datasets=args.max_datasets,
        )
        agg = aggregate_results(raw_results, metadata_list, flat_preds, flat_labels)
        all_results["synthetic"] = agg

        logger.info("\n=== SYNTHETIC TEST DATA RESULTS ===")
        logger.info("  Per-dataset average:")
        for method, stats in agg["overall"].items():
            logger.info(
                f"    {method:25s}: AUROC={stats['auroc_mean']:.4f}±{stats['auroc_std']:.4f} "
                f"AUPR={stats['aupr_mean']:.4f}±{stats['aupr_std']:.4f} "
                f"(n={stats['n_valid']})"
            )
        logger.info("  Pooled (flattened across all features):")
        for method, stats in agg["overall"].items():
            if "auroc_pooled" in stats:
                logger.info(
                    f"    {method:25s}: AUROC={stats['auroc_pooled']:.4f} "
                    f"AUPR={stats['aupr_pooled']:.4f} "
                    f"(n_features={stats['n_features_total']})"
                )
        logger.info("\n  By feature count:")
        for bname, bstats in agg["by_feature_count"].items():
            if bstats["n"] == 0:
                continue
            logger.info(f"    {bname} (n={bstats['n']}):")
            for method in ["interpretation_model", "random_forest", "gradient_boosting", "random"]:
                if method in bstats:
                    s = bstats[method]
                    logger.info(f"      {method:25s}: AUROC={s['auroc_mean']:.4f}±{s['auroc_std']:.4f}")

    # ─── DREAM4 evaluation ───
    if not args.skip_dream:
        logger.info("\n=== DREAM4 EVALUATION ===")
        dream_results = evaluate_on_dream4(model, device, args.n_estimators)
        all_results["dream4"] = dream_results

        for size_key, size_data in dream_results.items():
            logger.info(f"\n  {size_key} Summary:")
            for method, stats in size_data["summary"].items():
                logger.info(
                    f"    {method:25s}: AUROC={stats['auroc_mean']:.4f}±{stats['auroc_std']:.4f} "
                    f"AUPR={stats['aupr_mean']:.4f}±{stats['aupr_std']:.4f}"
                )

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rf_baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
