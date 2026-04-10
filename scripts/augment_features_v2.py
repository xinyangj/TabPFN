#!/usr/bin/env python
"""Re-extract enriched features from cached raw data using the updated SignalProcessor.

Reads raw data (X_train, y_train, X_test) and labels from an existing cache
directory, re-runs TabPFN inference, and processes signals with the current
(enriched) SignalProcessor to produce higher-dimensional feature vectors.

Output is written to a separate directory so the source cache is preserved.

Usage:
    python scripts/augment_features_v2.py --input-dir data/interpretation_cache_v2 \
                                          --output-dir data/interpretation_cache_v2_enriched
    # Resume after interruption (skips existing output files):
    python scripts/augment_features_v2.py --input-dir data/interpretation_cache_v2 \
                                          --output-dir data/interpretation_cache_v2_enriched
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_DIR = Path("data/interpretation_cache_v2")
OUTPUT_DIR = Path("data/interpretation_cache_v2_enriched")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CATEGORIES = [
    "between_features_attention",
    "between_items_attention",
    "embeddings",
    "mlp_activations",
    "gradients",
    "items_attention_gradients",
]


def enrich_cache(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    max_datasets: int = 0,
) -> None:
    """Re-extract enriched features for all datasets in input_dir."""
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(input_dir.glob("dataset_*.npz"))
    logger.info(f"Found {len(npz_files)} datasets in {input_dir}")

    if max_datasets > 0:
        npz_files = npz_files[:max_datasets]
        logger.info(f"Limiting to first {max_datasets} datasets")

    extractor = SignalExtractor(extract_gradients=True)
    processor = SignalProcessor()

    n_enriched = 0
    n_skipped = 0
    n_failed = 0
    t_total = time.time()

    for file_idx, npz_path in enumerate(npz_files):
        out_path = output_dir / npz_path.name
        if out_path.exists():
            n_skipped += 1
            continue

        try:
            data = np.load(npz_path, allow_pickle=False)
            keys = data.files

            # Require raw data
            if not all(k in keys for k in ("raw_X_train", "raw_y_train", "raw_X_test")):
                logger.warning(f"{npz_path.name}: missing raw data, skipping")
                data.close()
                n_failed += 1
                continue

            X_train = data["raw_X_train"]
            y_train = data["raw_y_train"]
            X_test = data["raw_X_test"]

            # Copy labels
            labels = {}
            for k in keys:
                if k.startswith("label_"):
                    labels[k] = data[k]

            data.close()

            # Fit TabPFN and extract signals
            reg = TabPFNRegressor(n_estimators=1, device=DEVICE)
            reg.fit(X_train, y_train)
            signals = extractor.extract(reg, X_train, y_train, X_test)

            # Process each category with the enriched processor
            category_vectors = {}
            parts_in_order = []
            for cat in CATEGORIES:
                try:
                    cat_fv = processor.process(signals, signal_categories={cat})
                    cat_fv = np.nan_to_num(cat_fv, nan=0.0, posinf=0.0, neginf=0.0)
                    category_vectors[cat] = cat_fv
                    parts_in_order.append(cat_fv)
                except (ValueError, Exception) as e:
                    logger.debug(f"{npz_path.name} category {cat} failed: {e}")

            if not parts_in_order:
                logger.warning(f"{npz_path.name}: no categories produced, skipping")
                n_failed += 1
                del reg, signals
                torch.cuda.empty_cache()
                continue

            feature_vectors = np.concatenate(parts_in_order, axis=1)
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)

            # Build output arrays
            arrays = {
                "feature_vectors": feature_vectors,
                "raw_X_train": X_train,
                "raw_y_train": y_train,
                "raw_X_test": X_test,
            }
            arrays.update(labels)
            for cat_name, cat_fv in category_vectors.items():
                arrays[f"cat_{cat_name}"] = cat_fv

            np.savez_compressed(out_path, **arrays)

            # Copy companion JSON metadata if present
            json_src = npz_path.with_suffix(".json")
            json_dst = out_path.with_suffix(".json")
            if json_src.exists() and not json_dst.exists():
                shutil.copy2(json_src, json_dst)

            n_enriched += 1
            del reg, signals
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"{npz_path.name} failed: {e}")
            n_failed += 1
            # Clean up partial output
            if out_path.exists():
                out_path.unlink()
            continue

        if n_enriched % 50 == 0 and n_enriched > 0:
            elapsed = time.time() - t_total
            rate = n_enriched / elapsed * 3600
            remaining = len(npz_files) - file_idx - 1
            eta_h = remaining / max(rate, 1)
            logger.info(
                f"[{n_enriched}/{len(npz_files)}] enriched, "
                f"{n_skipped} skipped, {n_failed} failed | "
                f"{elapsed/60:.1f}min elapsed | {rate:.0f}/hr | "
                f"ETA: {eta_h:.1f}h"
            )

    elapsed = time.time() - t_total
    logger.info(
        f"Done: {n_enriched} enriched, {n_skipped} skipped (already exist), "
        f"{n_failed} failed in {elapsed/60:.1f} min"
    )

    # Write manifest
    manifest = {
        "n_enriched": n_enriched,
        "n_skipped_existing": n_skipped,
        "n_failed": n_failed,
        "source_dir": str(input_dir),
        "elapsed_min": round(elapsed / 60, 1),
        "categories": CATEGORIES,
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {output_dir / 'manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract enriched features from cached raw data"
    )
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-datasets", type=int, default=0,
                        help="Max datasets to process (0=all)")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = args.device

    enrich_cache(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_datasets=args.max_datasets,
    )


if __name__ == "__main__":
    main()
