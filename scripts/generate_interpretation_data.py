#!/usr/bin/env python
"""Generate and cache interpretation training data to disk.

This script generates synthetic datasets using the external TabPFN prior
generator, runs TabPFN inference to extract internal signals, computes
ground-truth labels, and saves everything to disk for later training.

Each dataset is saved as a separate .npz file so that:
  - Generation can be resumed after interruption
  - Data can be generated in parallel across machines
  - Disk usage is predictable (~1.5 MB per dataset)

Usage:
    python scripts/generate_interpretation_data.py --n_datasets 10000 --seed 42
    python scripts/generate_interpretation_data.py --n_datasets 10000 --start_idx 5000  # resume
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

# ── Feature / sample distribution ────────────────────────────────────
# v2.5-aligned: Beta(0.95, 8.0) * [3, MAX_FEATURES]
MAX_FEATURES = 50
MAX_TRAIN_SAMPLES = 1000


def _sample_n_features(rng: np.random.Generator) -> int:
    beta_val = rng.beta(0.95, 8.0)
    return max(3, int(3 + beta_val * (MAX_FEATURES - 3)))


def _sample_n_samples(rng: np.random.Generator) -> int:
    return int(rng.integers(50, MAX_TRAIN_SAMPLES + 1))


# ── Save / load helpers ──────────────────────────────────────────────

def _signal_tensors_to_numpy(signals: dict) -> dict:
    """Convert torch tensors in a signals dict to numpy for serialization."""
    out: dict = {}
    for key, val in signals.items():
        if isinstance(val, torch.Tensor):
            out[key] = val.cpu().numpy()
        elif isinstance(val, dict):
            out[key] = _signal_tensors_to_numpy(val)
        else:
            out[key] = val
    return out


def save_dataset(path: Path, record: dict) -> None:
    """Save a single dataset record to an .npz file.

    Instead of storing raw (very large) attention tensors, we store
    per-category processed feature vectors.  This keeps each file small
    (~50 KB) while still enabling the ablation study.
    """
    fv = record["feature_vectors"]             # (n_features, D_total)
    labels = record["labels"]                  # dict of arrays
    metadata = record["metadata"]              # dict
    cat_vectors = record.get("category_vectors", {})  # per-category fvs

    arrays: dict[str, np.ndarray] = {
        "feature_vectors": fv,
    }

    # Labels
    for lm, arr in labels.items():
        arrays[f"label_{lm}"] = arr

    # Per-category feature vectors (for ablation)
    for cat_name, cat_fv in cat_vectors.items():
        arrays[f"cat_{cat_name}"] = cat_fv

    # Raw data (X_train, y_train, X_test) for future re-extraction
    raw_data = record.get("raw_data", {})
    for raw_name, raw_arr in raw_data.items():
        arrays[f"raw_{raw_name}"] = raw_arr

    # Save
    np.savez_compressed(path, **arrays)

    # Metadata as companion JSON (small, human-readable)
    meta_path = path.with_suffix(".json")
    # Make metadata JSON-serializable
    safe_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe_meta[k] = v
        elif isinstance(v, dict):
            safe_meta[k] = {str(kk): str(vv) for kk, vv in v.items()}
        elif isinstance(v, (list, tuple)):
            safe_meta[k] = [str(x) for x in v]
        else:
            safe_meta[k] = str(v)
    with open(meta_path, "w") as f:
        json.dump(safe_meta, f)


def _flatten_signals(d: dict, prefix: str, out: dict) -> None:
    """Recursively flatten a nested dict of arrays."""
    for key, val in d.items():
        full_key = f"{prefix}__{key}"
        if isinstance(val, np.ndarray):
            out[full_key] = val
        elif isinstance(val, dict):
            _flatten_signals(val, full_key, out)
        # Skip non-array, non-dict values (ints, strings, etc.)


def load_dataset(path: Path) -> dict:
    """Load a single cached dataset from disk."""
    data = np.load(path, allow_pickle=False)

    feature_vectors = data["feature_vectors"]

    labels = {}
    for key in data.files:
        if key.startswith("label_"):
            labels[key[len("label_"):]] = data[key]

    # Per-category feature vectors (for ablation)
    category_vectors = {}
    for key in data.files:
        if key.startswith("cat_"):
            category_vectors[key[len("cat_"):]] = data[key]

    # Raw data (X_train, y_train, X_test) if available
    raw_data = {}
    for key in data.files:
        if key.startswith("raw_"):
            raw_data[key[len("raw_"):]] = data[key]

    # Metadata
    meta_path = path.with_suffix(".json")
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    return {
        "feature_vectors": feature_vectors,
        "labels": labels,
        "metadata": metadata,
        "category_vectors": category_vectors,
        "raw_data": raw_data,
    }


def _unflatten_signals(data, prefix: str) -> dict:
    """Reconstruct nested dict from flat namespaced keys.

    Arrays are returned as ``torch.Tensor`` so that the signal processor
    (which uses torch operations) works unchanged.
    """
    result: dict = {}
    search = f"{prefix}__"
    for key in data.files:
        if not key.startswith(search):
            continue
        rest = key[len(search):]
        parts = rest.split("__")
        d = result
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        val = data[key]
        # Restore 0-d arrays to Python scalars
        if val.ndim == 0:
            val = val.item()
        else:
            # Convert arrays back to torch tensors for signal processor
            import torch
            val = torch.from_numpy(val)
        d[parts[-1]] = val
    return result


def load_all_datasets(cache_dir: Path | None = None) -> list[dict]:
    """Load all cached datasets from a directory."""
    if cache_dir is None:
        cache_dir = CACHE_DIR
    npz_files = sorted(cache_dir.glob("dataset_*.npz"))
    logger.info(f"Loading {len(npz_files)} datasets from {cache_dir}")
    datasets = []
    for p in npz_files:
        try:
            datasets.append(load_dataset(p))
        except Exception as e:
            logger.warning(f"Failed to load {p.name}: {e}")
    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


# ── Main generation loop ─────────────────────────────────────────────

def generate_and_cache(
    n_datasets: int = 10_000,
    seed: int = 42,
    start_idx: int = 0,
    cache_dir: Path | None = None,
) -> None:
    """Generate datasets and save each to disk individually."""
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
    from tabpfn.interpretation.synthetic_data.label_generator import compute_all_labels
    from tabpfn.interpretation.synthetic_data.tabpfn_prior_adapter import (
        TabPFNPriorAdapter,
    )

    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    gen = TabPFNPriorAdapter(seed=seed)
    extractor = SignalExtractor(extract_gradients=True)
    processor = SignalProcessor()
    rng = np.random.default_rng(seed)

    # Advance RNG to start_idx so we get the same draws for a given seed
    for _ in range(start_idx):
        _sample_n_features(rng)
        _sample_n_samples(rng)

    n_saved = 0
    n_skipped = 0
    t_total = time.time()
    timings = {"gen": [], "fit": [], "extract": [], "process": [], "save": []}

    for i in range(start_idx, start_idx + n_datasets):
        out_path = cache_dir / f"dataset_{i:06d}.npz"
        if out_path.exists():
            n_saved += 1
            # Still advance rng to keep determinism
            _sample_n_features(rng)
            _sample_n_samples(rng)
            continue

        n_feat = _sample_n_features(rng)
        n_samp = _sample_n_samples(rng)

        try:
            # Generate SCM data
            t0 = time.time()
            dataset = gen.generate(n_features=n_feat, n_samples=n_samp)
            timings["gen"].append(time.time() - t0)

            # Train/test split
            n_total = dataset.X.shape[0]
            n_train = min(max(int(n_total * 0.7), 20), MAX_TRAIN_SAMPLES)
            X_train = dataset.X[:n_train].astype(np.float32)
            X_test = dataset.X[n_train:].astype(np.float32)
            y_train = dataset.y[:n_train].astype(np.float32)
            if X_test.shape[0] == 0:
                X_test = dataset.X[-max(10, n_total // 5):].astype(np.float32)

            # Fit TabPFN
            t0 = time.time()
            reg = TabPFNRegressor(n_estimators=1, device=DEVICE)
            reg.fit(X_train, y_train)
            timings["fit"].append(time.time() - t0)

            # Extract signals
            t0 = time.time()
            signals = extractor.extract(reg, X_train, y_train, X_test)
            timings["extract"].append(time.time() - t0)

            # Process to feature vectors — full and per-category (single pass)
            t0 = time.time()
            category_vectors = {}
            parts_in_order = []
            CATEGORIES = [
                "between_features_attention",
                "between_items_attention",
                "embeddings",
                "mlp_activations",
                "gradients",
            ]
            for cat in CATEGORIES:
                try:
                    cat_fv = processor.process(signals, signal_categories={cat})
                    category_vectors[cat] = cat_fv
                    parts_in_order.append(cat_fv)
                except ValueError:
                    pass
            # Full feature vector = concat of all categories
            feature_vectors = np.concatenate(parts_in_order, axis=1)
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            timings["process"].append(time.time() - t0)

            # Compute labels
            labels = compute_all_labels(dataset)

            # Skip trivial datasets
            if labels["binary_direct"].sum() == 0 and labels["binary_ancestry"].sum() == 0:
                n_skipped += 1
                del reg, signals
                torch.cuda.empty_cache()
                continue

            record = {
                "feature_vectors": feature_vectors,
                "labels": labels,
                "metadata": dataset.metadata,
                "category_vectors": category_vectors,
                "raw_data": {
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_test": X_test,
                },
            }

            t0 = time.time()
            save_dataset(out_path, record)
            timings["save"].append(time.time() - t0)

            n_saved += 1
            del reg, signals, record
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Dataset {i} failed: {e}")
            continue

        if (n_saved) % 50 == 0:
            elapsed = time.time() - t_total
            rate = n_saved / elapsed * 3600
            logger.info(
                f"[{n_saved}/{n_datasets}] saved, {n_skipped} skipped | "
                f"{elapsed/60:.1f}min elapsed | {rate:.0f} datasets/hr | "
                f"Last: n_feat={n_feat}, n_samp={n_samp}, fv={feature_vectors.shape}"
            )

    elapsed = time.time() - t_total
    logger.info(
        f"Done: {n_saved} saved, {n_skipped} skipped in {elapsed/60:.1f} min"
    )
    if timings["gen"]:
        logger.info(
            f"Avg timings: gen={np.mean(timings['gen']):.2f}s "
            f"fit={np.mean(timings['fit']):.2f}s "
            f"extract={np.mean(timings['extract']):.2f}s "
            f"process={np.mean(timings['process']):.2f}s "
            f"save={np.mean(timings['save']):.2f}s"
        )

    # Write manifest
    manifest = {
        "n_saved": n_saved,
        "n_skipped": n_skipped,
        "seed": seed,
        "max_features": MAX_FEATURES,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "elapsed_min": round(elapsed / 60, 1),
        "generator": "tabpfn_prior (zzhang-cn/tabpfn-synthetic-data)",
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {cache_dir / 'manifest.json'}")


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate interpretation training data")
    parser.add_argument("--n_datasets", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0, help="Resume from this index")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device:
        DEVICE = args.device
    cache = Path(args.cache_dir) if args.cache_dir else CACHE_DIR

    generate_and_cache(
        n_datasets=args.n_datasets,
        seed=args.seed,
        start_idx=args.start_idx,
        cache_dir=cache,
    )
