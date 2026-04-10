#!/usr/bin/env python
"""Evaluate v6 interpretation model on DREAM4/5 GRN benchmarks.

Extracts features (1267 dims, 5 categories) for each target gene using the
GPU stats pipeline, then runs the trained interpretation model to predict
edge scores and evaluates against gold standards.

Usage:
    python scripts/eval_grn_v6.py [--device cuda:0]
    python scripts/eval_grn_v6.py --skip-extraction  # reuse cached features
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
from sklearn.metrics import average_precision_score, roc_auc_score

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# v6 model uses these 5 categories (1267 dims, enriched=False)
V6_CATEGORIES = sorted([
    "between_features_attention",
    "between_items_attention",
    "embeddings",
    "gradients",
    "mlp_activations",
])
V6_DIM = 1267

FEATURES_DIR = Path("data/dream_interpretation_features/v6")
MODEL_PATH = Path("results/interpretation_experiments/best_v6_600k_p30_model.pt")
OUT_DIR = Path("results/interpretation_experiments/grn_evaluation")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_target(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    target_gene: str,
    device: str,
) -> tuple[np.ndarray | None, list[str]]:
    """Extract v6-compatible features for one target gene.

    Uses GPU stats pipeline for speed and consistency with training data.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.gpu_stats_computer import GPUStatsComputer
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

    target_idx = gene_names.index(target_gene)

    # Exclude target from its own features (prevent data leakage)
    tf_indices = [gene_names.index(tf) for tf in tf_names if tf != target_gene]
    input_tf_names = [tf for tf in tf_names if tf != target_gene]

    X = expression[:, tf_indices].astype(np.float32)
    y = expression[:, target_idx].astype(np.float32)

    # 70/30 train/test split
    n_train = max(int(len(y) * 0.7), 20)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]

    if X_test.shape[0] == 0:
        X_test = X_train[-5:]

    # Train TabPFN
    reg = TabPFNRegressor(n_estimators=1, device=device)
    reg.fit(X_train, y_train)

    # Extract signals (input_only mode — matches training data)
    extractor = SignalExtractor(extract_gradients="input_only")
    try:
        signals = extractor.extract(reg, X_train, y_train, X_test)
    except Exception as e:
        logger.warning(f"  Signal extraction failed for {target_gene}: {e}")
        del reg
        torch.cuda.empty_cache()
        return None, input_tf_names

    # GPU stats → per-feature vectors (same pipeline as data generation)
    computer = GPUStatsComputer(enriched=False)
    processor = SignalProcessor(enriched=False)

    stats = computer.compute(signals)
    feature_vectors = processor.process_from_stats(
        stats, n_features=len(input_tf_names),
        signal_categories=set(V6_CATEGORIES),
    )

    feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)

    if feature_vectors.shape[1] != V6_DIM:
        logger.warning(
            f"  Dim mismatch for {target_gene}: got {feature_vectors.shape[1]}, expected {V6_DIM}"
        )

    del reg
    torch.cuda.empty_cache()

    return feature_vectors, input_tf_names


def extract_dream4_network(network_size: int, network_id: int, device: str) -> dict:
    """Extract features for all targets in one DREAM4 network."""
    from tabpfn.grn.datasets import DREAMChallengeLoader

    logger.info(f"Extracting DREAM4-{network_size} network {network_id}...")
    t0 = time.time()

    loader = DREAMChallengeLoader(data_path="data/dream4")
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=network_size, network_id=network_id,
    )
    logger.info(f"  Expression: {expression.shape}, TFs: {len(tf_names)}, gold edges: {len(gold_standard)}")

    target_genes = gene_names
    feature_vectors = {}
    input_tf_names = {}

    for i, target in enumerate(target_genes):
        logger.info(f"  [{i+1}/{len(target_genes)}] Target: {target}")
        fv, tfs = extract_features_for_target(
            expression, gene_names, tf_names, target, device,
        )
        if fv is not None:
            feature_vectors[target] = fv
            input_tf_names[target] = tfs
            logger.info(f"    shape={fv.shape}")

    elapsed = time.time() - t0
    logger.info(f"  Done: {len(feature_vectors)}/{len(target_genes)} targets, {elapsed:.1f}s")

    return {
        "gene_names": gene_names, "tf_names": tf_names,
        "target_genes": target_genes, "gold_standard": gold_standard,
        "feature_vectors": feature_vectors, "input_tf_names": input_tf_names,
        "time_s": elapsed,
    }


def extract_dream5_ecoli(device: str, max_targets: int = 0) -> dict:
    """Extract features for DREAM5 E. coli dataset."""
    logger.info("Extracting DREAM5 E. coli...")
    t0 = time.time()

    import pandas as pd
    expression = np.load("data/dream5/ecoli_expression.npy")
    genes_df = pd.read_csv("data/dream5/ecoli_genes.csv")
    tfs_df = pd.read_csv("data/dream5/ecoli_tfs.csv")

    # Gold standard may lack headers — detect and handle
    gold_df = pd.read_csv("data/dream5/ecoli_gold_standard.csv")
    if "weight" not in gold_df.columns:
        gold_df = pd.read_csv(
            "data/dream5/ecoli_gold_standard.csv",
            header=None, names=["tf", "target", "weight"],
        )

    gene_col = "gene" if "gene" in genes_df.columns else genes_df.columns[0]
    tf_col = "tf" if "tf" in tfs_df.columns else tfs_df.columns[0]
    gene_names = genes_df[gene_col].tolist()
    tf_names = tfs_df[tf_col].tolist()

    # Gold standard: only positive edges (weight == 1)
    gold_standard = gold_df[gold_df["weight"] == 1]

    logger.info(f"  Expression: {expression.shape}, genes: {len(gene_names)}, TFs: {len(tf_names)}")
    logger.info(f"  Gold edges: {len(gold_standard)}")

    # Target = all genes that appear as targets in gold standard
    all_targets = sorted(set(gold_df["target"].tolist()))
    # Filter to those in gene_names
    target_genes = [g for g in all_targets if g in gene_names]
    if max_targets > 0:
        target_genes = target_genes[:max_targets]
    logger.info(f"  Evaluating {len(target_genes)} target genes")

    feature_vectors = {}
    input_tf_names = {}

    for i, target in enumerate(target_genes):
        logger.info(f"  [{i+1}/{len(target_genes)}] Target: {target}")
        fv, tfs = extract_features_for_target(
            expression, gene_names, tf_names, target, device,
        )
        if fv is not None:
            feature_vectors[target] = fv
            input_tf_names[target] = tfs
            logger.info(f"    shape={fv.shape}")

    elapsed = time.time() - t0
    logger.info(f"  Done: {len(feature_vectors)}/{len(target_genes)} targets, {elapsed:.1f}s")

    return {
        "gene_names": gene_names, "tf_names": tf_names,
        "target_genes": target_genes, "gold_standard": gold_standard,
        "feature_vectors": feature_vectors, "input_tf_names": input_tf_names,
        "time_s": elapsed,
    }


def save_features(data: dict, out_path: Path):
    """Save extracted features to npz."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "gene_names": np.array(data["gene_names"]),
        "tf_names": np.array(data["tf_names"]),
        "target_genes": np.array(data["target_genes"]),
        "feature_dim": np.array(V6_DIM),
        "signal_categories": np.array(V6_CATEGORIES),
    }

    gs = data["gold_standard"]
    save_dict["gold_tf"] = np.array(gs["tf"].tolist())
    save_dict["gold_target"] = np.array(gs["target"].tolist())
    save_dict["gold_weight"] = np.array(gs["weight"].tolist())

    for target in data["target_genes"]:
        if target in data["feature_vectors"]:
            save_dict[f"fv_{target}"] = data["feature_vectors"][target]
            save_dict[f"tfs_{target}"] = np.array(data["input_tf_names"][target])

    np.savez_compressed(out_path, **save_dict)
    logger.info(f"  Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def load_features(features_path: Path) -> dict:
    """Load cached features from npz."""
    data = np.load(features_path, allow_pickle=True)

    gene_names = list(data["gene_names"])
    tf_names = list(data["tf_names"])
    target_genes = list(data["target_genes"])

    gold_tf = list(data["gold_tf"])
    gold_target = list(data["gold_target"])
    gold_weight = list(data["gold_weight"])
    gold_edges = set()
    for tf, tgt, w in zip(gold_tf, gold_target, gold_weight):
        if float(w) > 0:
            gold_edges.add((str(tf), str(tgt)))

    feature_vectors = {}
    input_tf_names = {}
    for target in target_genes:
        if f"fv_{target}" in data:
            feature_vectors[target] = data[f"fv_{target}"]
            input_tf_names[target] = list(data[f"tfs_{target}"])

    return {
        "gene_names": gene_names, "tf_names": tf_names,
        "target_genes": target_genes, "gold_edges": gold_edges,
        "feature_vectors": feature_vectors, "input_tf_names": input_tf_names,
    }


# ---------------------------------------------------------------------------
# Model loading and prediction
# ---------------------------------------------------------------------------

def load_v6_model(model_path: Path, device: str):
    """Load v6 interpretation model from checkpoint."""
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    model = InterpretationModel(
        variant="mlp",
        input_dim=ckpt["input_dim"],
        output_mode="binary",
        hidden_dims=cfg["hidden_dims"],
        dropout=cfg["dropout"],
        norm=cfg["norm"],
        activation=cfg["activation"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    global_mean = ckpt["global_mean"].to(device)
    global_std = ckpt["global_std"].to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded v6 model: {ckpt['input_dim']}d, {n_params:,} params")
    return model, global_mean, global_std


def predict_edge_scores(
    model: torch.nn.Module,
    global_mean: torch.Tensor,
    global_std: torch.Tensor,
    network_data: dict,
    device: str,
) -> dict[tuple[str, str], float]:
    """Predict edge scores using v6 interpretation model."""
    edge_scores = {}

    for target, fv in network_data["feature_vectors"].items():
        tf_names = network_data["input_tf_names"][target]
        fv_t = torch.from_numpy(fv.astype(np.float32)).to(device)

        # Apply z-score normalization (matching training pipeline)
        fv_t = (fv_t - global_mean) / global_std

        with torch.no_grad():
            logits = model(fv_t.unsqueeze(0))
            scores = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        for i, tf in enumerate(tf_names):
            edge_scores[(tf, target)] = float(scores[i])

    return edge_scores


def random_baseline(network_data: dict) -> dict[tuple[str, str], float]:
    """Random baseline."""
    rng = np.random.RandomState(42)
    edge_scores = {}
    for target, fv in network_data["feature_vectors"].items():
        for tf in network_data["input_tf_names"][target]:
            edge_scores[(tf, target)] = float(rng.random())
    return edge_scores


def attention_heuristic_baseline(network_data: dict) -> dict[tuple[str, str], float]:
    """Mean absolute activation as importance proxy."""
    edge_scores = {}
    for target, fv in network_data["feature_vectors"].items():
        tfs = network_data["input_tf_names"][target]
        scores = np.abs(fv).mean(axis=1)
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        for i, tf in enumerate(tfs):
            edge_scores[(tf, target)] = float(scores[i])
    return edge_scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_edges(
    edge_scores: dict[tuple[str, str], float],
    gold_edges: set[tuple[str, str]],
    all_possible_edges: set[tuple[str, str]] | None = None,
) -> dict[str, float]:
    """Evaluate edge predictions against gold standard."""
    if all_possible_edges is None:
        all_possible_edges = set(edge_scores.keys())

    y_true, y_score = [], []
    for edge in sorted(all_possible_edges):
        y_true.append(1 if edge in gold_edges else 0)
        y_score.append(edge_scores.get(edge, 0.0))

    y_true, y_score = np.array(y_true), np.array(y_score)
    n_pos = y_true.sum()

    if n_pos == 0 or n_pos == len(y_true):
        return {"auroc": 0.5, "aupr": float(n_pos / len(y_true)),
                "n_edges": int(n_pos), "n_total": len(y_true)}

    auroc = roc_auc_score(y_true, y_score)
    aupr = average_precision_score(y_true, y_score)

    k = int(n_pos)
    top_k = np.argsort(y_score)[::-1][:k]
    prec_at_k = y_true[top_k].sum() / k

    k10 = max(1, len(y_true) // 10)
    top10 = np.argsort(y_score)[::-1][:k10]
    ep10 = y_true[top10].sum() / k10

    return {
        "auroc": float(auroc), "aupr": float(aupr),
        "precision_at_k": float(prec_at_k),
        "early_precision_10pct": float(ep10),
        "n_edges": int(n_pos), "n_total": len(y_true),
        "edge_density": float(n_pos / len(y_true)),
    }


def evaluate_network(
    model, global_mean, global_std, network_data: dict, device: str,
) -> dict:
    """Evaluate all methods on one network."""
    all_edges = set()
    for target, tfs in network_data["input_tf_names"].items():
        for tf in tfs:
            all_edges.add((tf, target))

    gold = network_data["gold_edges"]
    results = {}

    # Interpretation model
    interp_scores = predict_edge_scores(model, global_mean, global_std, network_data, device)
    results["interpretation_model"] = evaluate_edges(interp_scores, gold, all_edges)

    # Random baseline
    results["random"] = evaluate_edges(random_baseline(network_data), gold, all_edges)

    # Attention heuristic
    results["attention_heuristic"] = evaluate_edges(
        attention_heuristic_baseline(network_data), gold, all_edges
    )

    for method, m in results.items():
        logger.info(f"  {method:25s}: AUROC={m['auroc']:.4f} AUPR={m['aupr']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate v6 model on DREAM4/5 GRN benchmarks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model", default=str(MODEL_PATH))
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip feature extraction, use cached features")
    parser.add_argument("--dream5-max-targets", type=int, default=0,
                        help="Max DREAM5 targets (0=all)")
    parser.add_argument("--skip-dream5", action="store_true")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    features_dir = FEATURES_DIR

    # ---- Feature extraction ----
    if not args.skip_extraction:
        logger.info("=" * 60)
        logger.info("Phase 1: Feature extraction (v6, 1267 dims)")
        logger.info("=" * 60)

        # DREAM4-10 (5 networks × 10 targets each)
        for nid in range(1, 6):
            data = extract_dream4_network(10, nid, device)
            save_features(data, features_dir / f"dream4_10_net{nid}.npz")

        # DREAM4-100 (5 networks × 100 targets each)
        for nid in range(1, 6):
            data = extract_dream4_network(100, nid, device)
            save_features(data, features_dir / f"dream4_100_net{nid}.npz")

        # DREAM5 E. coli
        if not args.skip_dream5:
            data = extract_dream5_ecoli(device, max_targets=args.dream5_max_targets)
            save_features(data, features_dir / "dream5_ecoli.npz")

    # ---- Load model ----
    logger.info("\n" + "=" * 60)
    logger.info("Phase 2: Model evaluation")
    logger.info("=" * 60)

    model, global_mean, global_std = load_v6_model(Path(args.model), device)
    all_results = {}

    # ---- DREAM4 evaluation ----
    for size in [10, 100]:
        size_results = []
        for nid in range(1, 6):
            fpath = features_dir / f"dream4_{size}_net{nid}.npz"
            if not fpath.exists():
                logger.warning(f"Missing: {fpath}")
                continue
            logger.info(f"\nDREAM4-{size} network {nid}:")
            nd = load_features(fpath)
            results = evaluate_network(model, global_mean, global_std, nd, device)
            size_results.append({
                "network": f"dream4_{size}_net{nid}",
                "n_targets": len(nd["feature_vectors"]),
                "n_gold_edges": len(nd["gold_edges"]),
                "results": results,
            })

        if size_results:
            methods = list(size_results[0]["results"].keys())
            agg = {}
            for method in methods:
                aurocs = [r["results"][method]["auroc"] for r in size_results]
                auprs = [r["results"][method]["aupr"] for r in size_results]
                agg[method] = {
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_std": float(np.std(aurocs)),
                    "aupr_mean": float(np.mean(auprs)),
                    "aupr_std": float(np.std(auprs)),
                }

            all_results[f"dream4_{size}"] = {
                "per_network": size_results,
                "aggregated": agg,
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"DREAM4-{size} Summary ({len(size_results)} networks)")
            logger.info(f"{'Method':<25} {'AUROC':>15} {'AUPR':>15}")
            logger.info("-" * 60)
            for method in methods:
                a = agg[method]
                logger.info(f"{method:<25} {a['auroc_mean']:.4f}±{a['auroc_std']:.4f} "
                            f"{a['aupr_mean']:.4f}±{a['aupr_std']:.4f}")

    # ---- DREAM5 evaluation ----
    if not args.skip_dream5:
        fpath = features_dir / "dream5_ecoli.npz"
        if fpath.exists():
            logger.info(f"\nDREAM5 E. coli:")
            nd = load_features(fpath)
            results = evaluate_network(model, global_mean, global_std, nd, device)
            all_results["dream5_ecoli"] = {
                "n_targets": len(nd["feature_vectors"]),
                "n_gold_edges": len(nd["gold_edges"]),
                "results": results,
            }

    # ---- Save results ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "grn_v6_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {out_path}")

    # ---- Print final comparison ----
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Dataset':<20} {'Method':<25} {'AUROC':>8} {'AUPR':>8}")
    logger.info("-" * 65)
    for dataset_key, dataset_val in all_results.items():
        if "aggregated" in dataset_val:
            for method, a in dataset_val["aggregated"].items():
                logger.info(f"{dataset_key:<20} {method:<25} {a['auroc_mean']:>8.4f} {a['aupr_mean']:>8.4f}")
        elif "results" in dataset_val:
            for method, m in dataset_val["results"].items():
                logger.info(f"{dataset_key:<20} {method:<25} {m['auroc']:>8.4f} {m['aupr']:>8.4f}")


if __name__ == "__main__":
    main()
