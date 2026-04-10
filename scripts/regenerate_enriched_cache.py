#!/usr/bin/env python
"""Regenerate interpretation cache with enriched signal features.

Loads raw data (X_train, y_train, X_test) from existing .npz files,
re-runs TabPFN inference, and recomputes signals with the enriched
signal processor (15 stats for between-features attention, 9 stats
for between-items attention).

Labels are preserved from the original files (no re-computation needed).

Usage:
    python scripts/regenerate_enriched_cache.py
    python scripts/regenerate_enriched_cache.py --start_idx 50000  # resume
    python scripts/regenerate_enriched_cache.py --output_dir data/interpretation_cache_enriched
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path("data/interpretation_cache")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LOG_FILE = Path("results/interpretation_experiments/regenerate_enriched.log")


def log(msg: str) -> None:
    """Write log message with flush."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(description="Regenerate enriched cache")
    parser.add_argument("--start_idx", type=int, default=0, help="Resume from this index")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: overwrite in-place)")
    parser.add_argument("--max_files", type=int, default=None, help="Process at most N files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import after arg parsing to avoid slow imports when just checking --help
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

    # Collect all existing dataset files
    files = sorted(CACHE_DIR.glob("dataset_*.npz"))
    log(f"Found {len(files)} dataset files in {CACHE_DIR}")

    if args.start_idx > 0:
        files = files[args.start_idx:]
        log(f"Resuming from index {args.start_idx}, {len(files)} files remaining")

    if args.max_files:
        files = files[:args.max_files]
        log(f"Processing at most {len(files)} files")

    # Initialize TabPFN and extraction pipeline
    log(f"Loading TabPFN on {DEVICE}...")
    reg = TabPFNRegressor(device=DEVICE)
    extractor = SignalExtractor(extract_gradients=False)
    processor = SignalProcessor()
    log("TabPFN loaded")

    CATEGORIES = [
        "between_features_attention",
        "between_items_attention",
        "embeddings",
        "mlp_activations",
    ]

    n_processed = 0
    n_skipped = 0
    n_errors = 0
    t_start = time.time()

    for i, fpath in enumerate(files):
        try:
            # Load existing data
            data = np.load(fpath, allow_pickle=False)

            # Check for raw data
            if "raw_X_train" not in data or "raw_y_train" not in data or "raw_X_test" not in data:
                n_skipped += 1
                if n_skipped % 100 == 0:
                    log(f"Skipped {n_skipped} files (no raw data)")
                continue

            X_train = data["raw_X_train"]
            y_train = data["raw_y_train"]
            X_test = data["raw_X_test"]

            # Preserve labels from original file
            labels_arrays = {}
            for key in data.files:
                if key.startswith("label_"):
                    labels_arrays[key] = data[key]

            # Re-run TabPFN inference
            reg.fit(X_train, y_train)
            signals = extractor.extract(reg, X_train, y_train, X_test)

            # Process with enriched signal processor
            category_vectors = {}
            parts_in_order = []
            for cat in CATEGORIES:
                try:
                    cat_fv = processor.process(signals, signal_categories={cat})
                    category_vectors[cat] = cat_fv
                    parts_in_order.append(cat_fv)
                except ValueError:
                    pass

            if not parts_in_order:
                n_skipped += 1
                continue

            feature_vectors = np.concatenate(parts_in_order, axis=1)
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)

            # Build output arrays
            out_arrays = {
                "feature_vectors": feature_vectors,
                "raw_X_train": X_train,
                "raw_y_train": y_train,
                "raw_X_test": X_test,
            }
            out_arrays.update(labels_arrays)
            for cat_name, cat_fv in category_vectors.items():
                out_arrays[f"cat_{cat_name}"] = cat_fv

            # Save (overwrite or to new dir)
            out_path = output_dir / fpath.name
            np.savez_compressed(out_path, **out_arrays)

            # Copy metadata JSON if it exists
            meta_src = fpath.with_suffix(".json")
            meta_dst = out_path.with_suffix(".json")
            if meta_src.exists() and meta_src != meta_dst:
                import shutil
                shutil.copy2(meta_src, meta_dst)

            n_processed += 1

            # Progress logging
            if n_processed % 100 == 0:
                elapsed = time.time() - t_start
                rate = n_processed / elapsed
                eta = (len(files) - i - 1) / rate if rate > 0 else 0
                log(f"Processed {n_processed}/{len(files)} "
                    f"(skipped={n_skipped}, errors={n_errors}) "
                    f"rate={rate:.1f}/s, ETA={eta/3600:.1f}h "
                    f"dims={feature_vectors.shape[1]}")

            # Memory cleanup
            del signals
            torch.cuda.empty_cache()

        except Exception as e:
            n_errors += 1
            if n_errors <= 20:
                log(f"ERROR on {fpath.name}: {e}")
            continue

    elapsed = time.time() - t_start
    log(f"DONE: processed={n_processed}, skipped={n_skipped}, errors={n_errors}, "
        f"time={elapsed/3600:.1f}h")


if __name__ == "__main__":
    main()
