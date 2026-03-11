#!/usr/bin/env python
"""Augment existing interpretation cache with gradient features.

For each cached .npz that lacks `cat_gradients`, this script:
1. Loads raw data (X_train, y_train, X_test) from the .npz if available
2. If raw data is not available, replays the generation using the same seed
3. Runs TabPFN inference with gradient extraction
4. Processes gradient signals into per-feature vectors
5. Appends `cat_gradients` to the .npz file

Usage:
    python scripts/augment_gradients.py [--cache-dir data/interpretation_cache]
                                        [--seed 42] [--max-datasets 0]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/interpretation_cache")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_FEATURES = 50
MAX_TRAIN_SAMPLES = 1000


def _sample_n_features(rng: np.random.Generator) -> int:
    beta_val = rng.beta(0.95, 8.0)
    return max(3, int(3 + beta_val * (MAX_FEATURES - 3)))


def _sample_n_samples(rng: np.random.Generator) -> int:
    return int(rng.integers(50, MAX_TRAIN_SAMPLES + 1))


def augment_with_gradients(
    cache_dir: Path = CACHE_DIR,
    seed: int = 42,
    max_datasets: int = 0,
) -> None:
    """Add gradient features to existing cached datasets."""
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

    cache_dir = Path(cache_dir)
    npz_files = sorted(cache_dir.glob("dataset_*.npz"))
    logger.info(f"Found {len(npz_files)} cached datasets in {cache_dir}")

    # Build index→file mapping from filenames (dataset_XXXXXX.npz)
    file_map: dict[int, Path] = {}
    for p in npz_files:
        try:
            idx = int(p.stem.split("_")[1])
            file_map[idx] = p
        except (ValueError, IndexError):
            continue

    if max_datasets > 0:
        # Limit to first N files
        indices = sorted(file_map.keys())[:max_datasets]
        file_map = {i: file_map[i] for i in indices}

    # Check how many already have gradients
    n_already = 0
    n_need = 0
    for p in file_map.values():
        data = np.load(p, allow_pickle=False)
        if "cat_gradients" in data.files:
            n_already += 1
        else:
            n_need += 1
        data.close()

    logger.info(f"Already augmented: {n_already}, need augmentation: {n_need}")
    if n_need == 0:
        logger.info("Nothing to do!")
        return

    extractor = SignalExtractor(extract_gradients=True)
    processor = SignalProcessor()

    # We need TabPFNPriorAdapter only for datasets without raw data
    gen = None  # Lazy init

    # Replay RNG to reproduce the same (n_feat, n_samp) draws per index
    rng = np.random.default_rng(seed)

    n_augmented = 0
    n_skipped = 0
    n_failed = 0
    t_total = time.time()

    # Process all indices in order (to keep RNG in sync for regeneration)
    all_indices = sorted(file_map.keys())
    max_idx = max(all_indices) if all_indices else 0

    # Pre-advance RNG state for indices and track draws
    rng_draws: dict[int, tuple[int, int]] = {}
    replay_rng = np.random.default_rng(seed)
    for i in range(max_idx + 1):
        n_feat = _sample_n_features(replay_rng)
        n_samp = _sample_n_samples(replay_rng)
        if i in file_map:
            rng_draws[i] = (n_feat, n_samp)

    for idx in all_indices:
        p = file_map[idx]

        try:
            data = np.load(p, allow_pickle=False)
            keys = data.files

            # Skip if already augmented
            if "cat_gradients" in keys:
                data.close()
                n_skipped += 1
                continue

            # Try to get raw data from file
            has_raw = "raw_X_train" in keys and "raw_y_train" in keys and "raw_X_test" in keys
            if has_raw:
                X_train = data["raw_X_train"]
                y_train = data["raw_y_train"]
                X_test = data["raw_X_test"]
            else:
                # Must regenerate the data
                if gen is None:
                    from tabpfn.interpretation.synthetic_data.tabpfn_prior_adapter import (
                        TabPFNPriorAdapter,
                    )
                    gen = TabPFNPriorAdapter(seed=seed)

                n_feat, n_samp = rng_draws[idx]
                dataset = gen.generate(n_features=n_feat, n_samples=n_samp)

                n_total = dataset.X.shape[0]
                n_train = min(max(int(n_total * 0.7), 20), MAX_TRAIN_SAMPLES)
                X_train = dataset.X[:n_train].astype(np.float32)
                X_test = dataset.X[n_train:].astype(np.float32)
                y_train = dataset.y[:n_train].astype(np.float32)
                if X_test.shape[0] == 0:
                    X_test = dataset.X[-max(10, n_total // 5):].astype(np.float32)

            # Fit TabPFN and extract signals with gradients
            reg = TabPFNRegressor(n_estimators=1, device=DEVICE)
            reg.fit(X_train, y_train)
            signals = extractor.extract(reg, X_train, y_train, X_test)

            # Process only gradient category
            grad_fv = processor.process(signals, signal_categories={"gradients"})
            grad_fv = np.nan_to_num(grad_fv, nan=0.0, posinf=0.0, neginf=0.0)

            # Rebuild the .npz with the new cat_gradients key
            existing_arrays = {k: data[k] for k in keys}
            existing_arrays["cat_gradients"] = grad_fv

            # Also update feature_vectors to include gradient dims
            # New full = old full + gradient dims
            old_fv = data["feature_vectors"]
            new_fv = np.concatenate([old_fv, grad_fv], axis=1)
            existing_arrays["feature_vectors"] = new_fv

            # Store raw data if not already present
            if not has_raw:
                existing_arrays["raw_X_train"] = X_train.astype(np.float32)
                existing_arrays["raw_y_train"] = y_train.astype(np.float32)
                existing_arrays["raw_X_test"] = X_test.astype(np.float32)

            data.close()

            np.savez_compressed(p, **existing_arrays)

            n_augmented += 1
            del reg, signals
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Dataset {idx} ({p.name}) failed: {e}")
            n_failed += 1
            continue

        if n_augmented % 50 == 0 and n_augmented > 0:
            elapsed = time.time() - t_total
            rate = n_augmented / elapsed * 3600
            eta_h = (n_need - n_augmented) / max(rate, 1)
            logger.info(
                f"[{n_augmented}/{n_need}] augmented, {n_failed} failed | "
                f"{elapsed/60:.1f}min elapsed | {rate:.0f}/hr | "
                f"ETA: {eta_h:.1f}h"
            )

    elapsed = time.time() - t_total
    logger.info(
        f"Done: {n_augmented} augmented, {n_skipped} skipped, {n_failed} failed "
        f"in {elapsed/60:.1f} min"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment cache with gradient features")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-datasets", type=int, default=0,
                        help="Max datasets to process (0=all)")
    args = parser.parse_args()

    augment_with_gradients(
        cache_dir=args.cache_dir,
        seed=args.seed,
        max_datasets=args.max_datasets,
    )


if __name__ == "__main__":
    main()
