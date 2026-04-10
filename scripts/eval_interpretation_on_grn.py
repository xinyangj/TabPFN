#!/usr/bin/env python
"""Evaluate interpretation model on DREAM4 GRN benchmarks.

Loads cached features (from extract_dream_features.py) and a trained
interpretation model checkpoint, then evaluates edge prediction accuracy
against the DREAM4 gold standard.

Usage:
    python scripts/eval_interpretation_on_grn.py [--device cuda:1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/dream_interpretation_features")
MODEL_PATH = Path("results/interpretation_experiments/best_v2_model.pt")
OUT_DIR = Path("results/interpretation_experiments/grn_evaluation")


def load_model(model_path: Path, device: str):
    """Load interpretation model from checkpoint."""
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
    logger.info(
        f"Loaded model: {checkpoint['input_dim']} dims, "
        f"test AUROC={checkpoint['test_metrics']['auroc']:.4f}"
    )
    return model


def load_network_features(features_path: Path) -> dict:
    """Load cached features for one DREAM4 network."""
    data = np.load(features_path, allow_pickle=True)

    gene_names = list(data["gene_names"])
    tf_names = list(data["tf_names"])
    target_genes = list(data["target_genes"])

    # Reconstruct gold standard
    gold_tf = list(data["gold_tf"])
    gold_target = list(data["gold_target"])
    gold_weight = list(data["gold_weight"])
    gold_edges = set()
    for tf, tgt, w in zip(gold_tf, gold_target, gold_weight):
        if float(w) > 0:
            gold_edges.add((str(tf), str(tgt)))

    # Load per-target features
    feature_vectors = {}
    input_tf_names = {}
    for target in target_genes:
        fv_key = f"fv_{target}"
        tfs_key = f"tfs_{target}"
        if fv_key in data:
            feature_vectors[target] = data[fv_key]
            input_tf_names[target] = list(data[tfs_key])

    return {
        "gene_names": gene_names,
        "tf_names": tf_names,
        "target_genes": target_genes,
        "gold_edges": gold_edges,
        "feature_vectors": feature_vectors,
        "input_tf_names": input_tf_names,
    }


def predict_edge_scores(
    model: torch.nn.Module,
    network_data: dict,
    device: str,
) -> dict[tuple[str, str], float]:
    """Run interpretation model on all targets to get edge scores.

    Returns
    -------
    dict of (tf, target) -> score
    """
    edge_scores = {}

    for target_gene, fv in network_data["feature_vectors"].items():
        tf_names = network_data["input_tf_names"][target_gene]
        fv_tensor = torch.from_numpy(fv.astype(np.float32)).to(device)

        # Add batch dim: (1, n_tfs, D)
        with torch.no_grad():
            logits = model(fv_tensor.unsqueeze(0))
            scores = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        for i, tf_name in enumerate(tf_names):
            edge_scores[(tf_name, target_gene)] = float(scores[i])

    return edge_scores


def random_baseline(network_data: dict) -> dict[tuple[str, str], float]:
    """Random baseline: uniform random scores."""
    rng = np.random.RandomState(42)
    edge_scores = {}
    for target_gene, fv in network_data["feature_vectors"].items():
        tf_names = network_data["input_tf_names"][target_gene]
        for tf_name in tf_names:
            edge_scores[(tf_name, target_gene)] = float(rng.random())
    return edge_scores


def attention_heuristic_baseline(network_data: dict) -> dict[tuple[str, str], float]:
    """Attention heuristic: use mean of feature vector as score.

    This is a simple baseline that uses the average activation
    across all signal dimensions as an importance proxy.
    """
    edge_scores = {}
    for target_gene, fv in network_data["feature_vectors"].items():
        tf_names = network_data["input_tf_names"][target_gene]
        # Use absolute mean of feature vector as importance score
        mean_scores = np.abs(fv).mean(axis=1)
        # Normalize to [0, 1]
        if mean_scores.max() > mean_scores.min():
            mean_scores = (mean_scores - mean_scores.min()) / (mean_scores.max() - mean_scores.min())
        for i, tf_name in enumerate(tf_names):
            edge_scores[(tf_name, target_gene)] = float(mean_scores[i])
    return edge_scores


def evaluate_edges(
    edge_scores: dict[tuple[str, str], float],
    gold_edges: set[tuple[str, str]],
    all_possible_edges: set[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """Evaluate edge predictions against gold standard.

    Parameters
    ----------
    edge_scores : dict of (tf, target) -> score
    gold_edges : set of (tf, target) tuples that are true edges
    all_possible_edges : set of all possible (tf, target) pairs

    Returns
    -------
    dict with AUROC, AUPR, and other metrics
    """
    if all_possible_edges is None:
        all_possible_edges = set(edge_scores.keys())

    # Build label and score arrays for all possible edges
    y_true = []
    y_score = []
    for edge in sorted(all_possible_edges):
        y_true.append(1 if edge in gold_edges else 0)
        y_score.append(edge_scores.get(edge, 0.0))

    y_true = np.array(y_true)
    y_score = np.array(y_score)

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        logger.warning("  Degenerate: all labels are the same")
        return {"auroc": 0.5, "aupr": n_pos / len(y_true), "n_edges": int(n_pos), "n_total": len(y_true)}

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    # Precision at k (k = number of true edges)
    k = int(n_pos)
    top_k_indices = np.argsort(y_score)[::-1][:k]
    precision_at_k = y_true[top_k_indices].sum() / k

    # Early precision (top 10%)
    k10 = max(1, len(y_true) // 10)
    top10_indices = np.argsort(y_score)[::-1][:k10]
    ep10 = y_true[top10_indices].sum() / k10

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "precision_at_k": float(precision_at_k),
        "early_precision_10pct": float(ep10),
        "n_edges": int(n_pos),
        "n_total": len(y_true),
        "edge_density": float(n_pos / len(y_true)),
    }


def evaluate_network(
    model: torch.nn.Module,
    features_path: Path,
    device: str,
) -> dict:
    """Evaluate all methods on one DREAM4 network."""
    network_data = load_network_features(features_path)

    logger.info(
        f"  Loaded: {len(network_data['feature_vectors'])} targets, "
        f"{len(network_data['gold_edges'])} gold edges"
    )

    # Get all possible edges (for consistent evaluation)
    all_edges = set()
    for target, tfs in network_data["input_tf_names"].items():
        for tf in tfs:
            all_edges.add((tf, target))

    results = {}

    # 1. Interpretation model
    interp_scores = predict_edge_scores(model, network_data, device)
    results["interpretation_model"] = evaluate_edges(
        interp_scores, network_data["gold_edges"], all_edges
    )
    logger.info(
        f"  Interpretation: AUROC={results['interpretation_model']['auroc']:.4f}, "
        f"AUPR={results['interpretation_model']['aupr']:.4f}"
    )

    # 2. Random baseline
    random_scores = random_baseline(network_data)
    results["random"] = evaluate_edges(
        random_scores, network_data["gold_edges"], all_edges
    )
    logger.info(
        f"  Random:         AUROC={results['random']['auroc']:.4f}, "
        f"AUPR={results['random']['aupr']:.4f}"
    )

    # 3. Attention heuristic
    attn_scores = attention_heuristic_baseline(network_data)
    results["attention_heuristic"] = evaluate_edges(
        attn_scores, network_data["gold_edges"], all_edges
    )
    logger.info(
        f"  Attn heuristic: AUROC={results['attention_heuristic']['auroc']:.4f}, "
        f"AUPR={results['attention_heuristic']['aupr']:.4f}"
    )

    return {
        "network": features_path.stem,
        "n_targets": len(network_data["feature_vectors"]),
        "n_gold_edges": len(network_data["gold_edges"]),
        "n_possible_edges": len(all_edges),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--features-dir", default=str(FEATURES_DIR))
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100])
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    model = load_model(Path(args.model), device)

    # Find all feature files
    features_dir = Path(args.features_dir)
    all_results = {}

    for size in args.sizes:
        size_results = []
        for nid in range(1, 6):
            fpath = features_dir / f"dream4_{size}_net{nid}.npz"
            if not fpath.exists():
                logger.warning(f"Features not found: {fpath}")
                continue

            logger.info(f"\nEvaluating DREAM4-{size} network {nid}...")
            result = evaluate_network(model, fpath, device)
            size_results.append(result)

        if size_results:
            # Aggregate across networks
            methods = list(size_results[0]["results"].keys())
            agg = {}
            for method in methods:
                aurocs = [r["results"][method]["auroc"] for r in size_results]
                auprs = [r["results"][method]["aupr"] for r in size_results]
                p_at_k = [r["results"][method]["precision_at_k"] for r in size_results]
                agg[method] = {
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "aupr_mean": float(np.mean(auprs)),
                    "aupr_std": float(np.std(auprs)),
                    "precision_at_k_mean": float(np.mean(p_at_k)),
                    "per_network": {
                        f"net{i+1}": {"auroc": aurocs[i], "aupr": auprs[i]}
                        for i in range(len(aurocs))
                    },
                }

            all_results[f"dream4_{size}"] = {
                "per_network": size_results,
                "aggregated": agg,
            }

            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info(f"DREAM4-{size} Summary (across {len(size_results)} networks)")
            logger.info(f"{'Method':<25} {'AUROC':>12} {'AUPR':>12} {'P@k':>12}")
            logger.info("-" * 65)
            for method in methods:
                a = agg[method]
                logger.info(
                    f"{method:<25} "
                    f"{a['auroc_mean']:.4f}±{a['auroc_std']:.4f} "
                    f"{a['aupr_mean']:.4f}±{a['aupr_std']:.4f} "
                    f"{a['precision_at_k_mean']:.4f}"
                )

    # Save results
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "dream4_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
