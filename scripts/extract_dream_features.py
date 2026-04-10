#!/usr/bin/env python
"""Extract interpretation features from DREAM4 benchmark datasets.

For each target gene:
1. Train TabPFN regressor (predict target from TF expression)
2. Extract internal signals (attention, MLP activations, gradients)
3. Process into per-feature vectors matching training categories (691 dims)

Saves cached features per network for fast evaluation with interpretation model.

Usage:
    python scripts/extract_dream_features.py [--device cuda:1] [--sizes 10 100]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = Path("data/dream_interpretation_features")

# V2-compatible 4 categories (691 dims)
V2_CATEGORIES = sorted([
    "between_features_attention",
    "between_items_attention",
    "mlp_activations",
    "gradients",
])

# All 6 categories (1591 dims) — matches v3 training
ALL_CATEGORIES = sorted([
    "between_features_attention",
    "between_items_attention",
    "embeddings",
    "mlp_activations",
    "gradients",
    "items_attention_gradients",
])

CATEGORY_SETS = {
    "v2": (V2_CATEGORIES, 691),
    "all": (ALL_CATEGORIES, 1591),
}


def load_dream4_data(network_size: int, network_id: int):
    """Load DREAM4 dataset using the existing loader."""
    from tabpfn.grn.datasets import DREAMChallengeLoader

    loader = DREAMChallengeLoader(data_path="data/dream4")
    expression, gene_names, tf_names, gold_standard = loader.load_dream4(
        network_size=network_size, network_id=network_id
    )
    return expression, gene_names, tf_names, gold_standard


def extract_features_for_target(
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    target_gene: str,
    device: str,
    signal_categories: list[str] | None = None,
    expected_dim: int = 691,
) -> np.ndarray | None:
    """Extract interpretation features for one target gene.

    Returns
    -------
    np.ndarray or None
        Per-TF feature matrix of shape (n_tfs, expected_dim), or None on failure.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

    if signal_categories is None:
        signal_categories = V2_CATEGORIES

    # Determine gradient extraction mode based on whether we need attention gradients
    needs_attn_grads = "items_attention_gradients" in signal_categories
    extract_gradients = True if needs_attn_grads else "input_only"

    target_idx = gene_names.index(target_gene)

    # Build feature matrix: TF expression columns (excluding target if it's a TF)
    tf_indices = [gene_names.index(tf) for tf in tf_names if tf != target_gene]
    input_tf_names = [tf for tf in tf_names if tf != target_gene]

    X = expression[:, tf_indices].astype(np.float32)
    y = expression[:, target_idx].astype(np.float32)

    # 70/30 train/test split (matching training data protocol)
    n_train = max(int(len(y) * 0.7), 20)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train = y[:n_train]

    if X_test.shape[0] == 0:
        X_test = X_train[-5:]  # Use last 5 train samples as test fallback

    # Train TabPFN
    reg = TabPFNRegressor(n_estimators=1, device=device)
    reg.fit(X_train, y_train)

    # Extract signals
    extractor = SignalExtractor(extract_gradients=extract_gradients)
    try:
        signals = extractor.extract(reg, X_train, y_train, X_test)
    except Exception as e:
        logger.warning(f"  Signal extraction failed for {target_gene}: {e}")
        return None, input_tf_names

    # Process into per-feature vectors
    processor = SignalProcessor(enriched=False)
    parts = []
    for cat in signal_categories:
        try:
            cat_fv = processor.process(signals, signal_categories={cat})
            parts.append(cat_fv)
        except Exception as e:
            logger.warning(f"  Category '{cat}' failed for {target_gene}: {e}")
            return None, input_tf_names

    feature_vectors = np.concatenate(parts, axis=1)
    feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)

    if feature_vectors.shape[1] != expected_dim:
        logger.warning(
            f"  Dimension mismatch for {target_gene}: got {feature_vectors.shape[1]}, expected {expected_dim}"
        )

    # Clean up GPU memory
    del reg
    torch.cuda.empty_cache()

    return feature_vectors, input_tf_names


def extract_network(
    network_size: int, network_id: int, device: str,
    signal_categories: list[str] | None = None,
    expected_dim: int = 691,
) -> dict:
    """Extract features for all targets in one DREAM4 network."""
    logger.info(f"Processing DREAM4-{network_size} network {network_id}...")
    t0 = time.time()

    expression, gene_names, tf_names, gold_standard = load_dream4_data(
        network_size, network_id
    )
    logger.info(
        f"  Expression: {expression.shape}, genes: {len(gene_names)}, "
        f"TFs: {len(tf_names)}, gold edges: {len(gold_standard)}"
    )

    # For DREAM4, all genes can be targets
    target_genes = gene_names

    all_feature_vectors = {}  # target_name -> (n_tfs, D) array
    all_tf_names = {}  # target_name -> list of TF names (excluding self)
    failed_targets = []

    for i, target_gene in enumerate(target_genes):
        logger.info(f"  [{i+1}/{len(target_genes)}] Target: {target_gene}")
        result = extract_features_for_target(
            expression, gene_names, tf_names, target_gene, device,
            signal_categories=signal_categories,
            expected_dim=expected_dim,
        )
        fv, input_tfs = result

        if fv is not None:
            all_feature_vectors[target_gene] = fv
            all_tf_names[target_gene] = input_tfs
            logger.info(f"    Features: {fv.shape}, dim={fv.shape[1]}")
        else:
            failed_targets.append(target_gene)
            logger.warning(f"    FAILED for {target_gene}")

    elapsed = time.time() - t0
    logger.info(
        f"  Done: {len(all_feature_vectors)}/{len(target_genes)} targets, "
        f"{len(failed_targets)} failed, {elapsed:.1f}s"
    )

    return {
        "network_size": network_size,
        "network_id": network_id,
        "gene_names": gene_names,
        "tf_names": tf_names,
        "target_genes": target_genes,
        "feature_vectors": all_feature_vectors,
        "input_tf_names": all_tf_names,
        "gold_standard": gold_standard,
        "feature_dim": expected_dim,
        "signal_categories": signal_categories if signal_categories else V2_CATEGORIES,
        "failed_targets": failed_targets,
        "extraction_time_s": elapsed,
    }


def save_network_features(data: dict, out_dir: Path) -> Path:
    """Save extracted features for one network to npz file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = data["network_size"]
    nid = data["network_id"]
    out_path = out_dir / f"dream4_{size}_net{nid}.npz"

    # Prepare arrays for saving
    save_dict = {
        "network_size": np.array(size),
        "network_id": np.array(nid),
        "gene_names": np.array(data["gene_names"]),
        "tf_names": np.array(data["tf_names"]),
        "target_genes": np.array(data["target_genes"]),
        "feature_dim": np.array(data["feature_dim"]),
        "signal_categories": np.array(data["signal_categories"]),
    }

    # Save gold standard as structured arrays
    gs = data["gold_standard"]
    save_dict["gold_tf"] = np.array(gs["tf"].tolist())
    save_dict["gold_target"] = np.array(gs["target"].tolist())
    save_dict["gold_weight"] = np.array(gs["weight"].tolist())

    # Save per-target feature vectors and TF names
    for target_gene in data["target_genes"]:
        if target_gene in data["feature_vectors"]:
            fv = data["feature_vectors"][target_gene]
            save_dict[f"fv_{target_gene}"] = fv
            save_dict[f"tfs_{target_gene}"] = np.array(
                data["input_tf_names"][target_gene]
            )

    np.savez_compressed(out_path, **save_dict)
    logger.info(f"Saved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100])
    parser.add_argument("--networks", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--categories", default="all", choices=["v2", "all"])
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    signal_categories, expected_dim = CATEGORY_SETS[args.categories]
    out_dir = Path(args.output_dir) if args.output_dir else OUT_DIR / args.categories

    logger.info(f"Using device: {device}")
    logger.info(f"Network sizes: {args.sizes}, IDs: {args.networks}")
    logger.info(f"Categories ({args.categories}): {signal_categories}")
    logger.info(f"Expected feature dim: {expected_dim}")
    logger.info(f"Output dir: {out_dir}")

    total_t0 = time.time()
    results_summary = []

    for size in args.sizes:
        for nid in args.networks:
            data = extract_network(
                size, nid, device,
                signal_categories=signal_categories,
                expected_dim=expected_dim,
            )
            out_path = save_network_features(data, out_dir)
            results_summary.append({
                "network": f"dream4_{size}_net{nid}",
                "targets_ok": len(data["feature_vectors"]),
                "targets_failed": len(data["failed_targets"]),
                "feature_dim": data["feature_dim"],
                "time_s": data["extraction_time_s"],
                "path": str(out_path),
            })

    total_time = time.time() - total_t0
    logger.info(f"\n{'='*60}")
    logger.info(f"All extractions complete in {total_time:.0f}s")
    for r in results_summary:
        logger.info(f"  {r['network']}: {r['targets_ok']} targets, dim={r['feature_dim']}, {r['time_s']:.0f}s")


if __name__ == "__main__":
    main()
