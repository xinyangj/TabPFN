#!/usr/bin/env python
"""Generate interpretation training data v4 — optimized with pipelining + mixed precision.

Improvements over v3:
  - CPU/GPU pipelining: prefetch next dataset on CPU while GPU processes current
  - Mixed precision: torch.amp.autocast(bfloat16) for forward/backward pass
  - Flash attention: always uses SDPA (straight-through for gradient extraction)
  - Feature range: Uniform[10, 150] (same as v3)
  - Output: 1591-dim feature vectors (6 categories including items_attention_gradients)

Usage:
    # Single GPU
    python scripts/generate_interpretation_data_v4.py --n_datasets 1000 --device cuda:0

    # Multi-GPU parallel (run in separate terminals / nohup)
    python scripts/generate_interpretation_data_v4.py --n_datasets 1000 --start_idx 0    --seed 100 --device cuda:0
    python scripts/generate_interpretation_data_v4.py --n_datasets 1000 --start_idx 1000 --seed 200 --device cuda:1
    python scripts/generate_interpretation_data_v4.py --n_datasets 1000 --start_idx 2000 --seed 300 --device cuda:2
    python scripts/generate_interpretation_data_v4.py --n_datasets 1000 --start_idx 3000 --seed 400 --device cuda:3
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/interpretation_cache_v4")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MIN_FEATURES = 10
MAX_FEATURES = 150
MAX_TRAIN_SAMPLES = 1000
PREFETCH_QUEUE_SIZE = 2


def _sample_n_features(rng: np.random.Generator) -> int:
    return int(rng.integers(MIN_FEATURES, MAX_FEATURES + 1))


def _sample_n_samples(rng: np.random.Generator) -> int:
    return int(rng.integers(50, MAX_TRAIN_SAMPLES + 1))


def save_dataset(path: Path, record: dict) -> None:
    """Save a single dataset record to an .npz file."""
    fv = record["feature_vectors"]
    labels = record["labels"]
    metadata = record["metadata"]
    cat_vectors = record.get("category_vectors", {})

    arrays: dict[str, np.ndarray] = {
        "feature_vectors": fv,
    }

    for lm, arr in labels.items():
        arrays[f"label_{lm}"] = arr

    for cat_name, cat_fv in cat_vectors.items():
        arrays[f"cat_{cat_name}"] = cat_fv

    raw_data = record.get("raw_data", {})
    for raw_name, raw_arr in raw_data.items():
        arrays[f"raw_{raw_name}"] = raw_arr

    np.savez_compressed(path, **arrays)

    meta_path = path.with_suffix(".json")
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


# ── CPU prefetch worker ──────────────────────────────────────────────

def _prefetch_worker(
    gen,
    rng: np.random.Generator,
    data_queue: queue.Queue,
    index_iter,
    cache_dir: Path,
    force: bool,
):
    """Generate datasets on CPU and push to queue for GPU processing."""
    for i in index_iter:
        out_path = cache_dir / f"dataset_{i:06d}.npz"
        n_feat = _sample_n_features(rng)
        n_samp = _sample_n_samples(rng)

        if out_path.exists() and not force:
            data_queue.put(("skip", i, out_path, None, None, None))
            continue

        try:
            t0 = time.time()
            dataset = gen.generate(n_features=n_feat, n_samples=n_samp)
            gen_time = time.time() - t0

            n_total = dataset.X.shape[0]
            n_train = min(max(int(n_total * 0.7), 20), MAX_TRAIN_SAMPLES)
            X_train = dataset.X[:n_train].astype(np.float32)
            X_test = dataset.X[n_train:].astype(np.float32)
            y_train = dataset.y[:n_train].astype(np.float32)
            if X_test.shape[0] == 0:
                X_test = dataset.X[-max(10, n_total // 5):].astype(np.float32)

            data_queue.put((
                "data", i, out_path, dataset,
                (X_train, y_train, X_test),
                {"gen_time": gen_time, "n_feat": n_feat, "n_samp": n_samp},
            ))
        except Exception as e:
            logger.warning(f"Dataset {i} generation failed: {e}")
            data_queue.put(("error", i, out_path, None, None, None))

    data_queue.put(("done", -1, None, None, None, None))


# ── Main generation loop with pipelining ─────────────────────────────

def generate_and_cache(
    n_datasets: int = 1000,
    seed: int = 100,
    start_idx: int = 0,
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    use_amp: bool = True,
) -> None:
    """Generate datasets with CPU/GPU pipelining and mixed precision."""
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
    extractor = SignalExtractor(extract_gradients="input_only")
    processor = SignalProcessor(enriched=False)
    rng = np.random.default_rng(seed)

    # Advance RNG to start_idx
    for _ in range(start_idx):
        _sample_n_features(rng)
        _sample_n_samples(rng)

    CATEGORIES = [
        "between_features_attention",
        "between_items_attention",
        "embeddings",
        "mlp_activations",
        "gradients",
        "items_attention_gradients",
    ]

    # Start CPU prefetch thread
    data_queue: queue.Queue = queue.Queue(maxsize=PREFETCH_QUEUE_SIZE)
    index_iter = range(start_idx, start_idx + n_datasets)
    prefetch_thread = threading.Thread(
        target=_prefetch_worker,
        args=(gen, rng, data_queue, index_iter, cache_dir, force),
        daemon=True,
    )
    prefetch_thread.start()

    n_saved = 0
    n_skipped = 0
    t_total = time.time()
    timings = {"gen": [], "fit": [], "extract": [], "process": [], "save": [], "total_gpu": []}

    amp_enabled = use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if amp_enabled:
        logger.info("Mixed precision enabled (bfloat16)")
    else:
        logger.info("Mixed precision disabled")

    while True:
        msg_type, idx, out_path, dataset, data_tuple, meta = data_queue.get()

        if msg_type == "done":
            break
        elif msg_type == "skip":
            n_saved += 1
            continue
        elif msg_type == "error":
            n_skipped += 1
            continue

        # msg_type == "data"
        X_train, y_train, X_test = data_tuple
        gen_time = meta["gen_time"]
        n_feat = meta["n_feat"]
        n_samp = meta["n_samp"]
        timings["gen"].append(gen_time)

        try:
            t_gpu_start = time.time()

            # Fit TabPFN
            t0 = time.time()
            reg = TabPFNRegressor(n_estimators=1, device=DEVICE)
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    reg.fit(X_train, y_train)
            else:
                reg.fit(X_train, y_train)
            timings["fit"].append(time.time() - t0)

            # Extract signals with optional mixed precision
            t0 = time.time()
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    signals = extractor.extract(reg, X_train, y_train, X_test)
            else:
                signals = extractor.extract(reg, X_train, y_train, X_test)
            # Ensure all tensors are float32 before CPU processing
            for key, val in signals.items():
                if isinstance(val, torch.Tensor):
                    signals[key] = val.float()
                elif isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, torch.Tensor):
                            val[k] = v.float()
            timings["extract"].append(time.time() - t0)

            # Process to feature vectors
            t0 = time.time()
            category_vectors = {}
            parts_in_order = []
            for cat in CATEGORIES:
                try:
                    cat_fv = processor.process(signals, signal_categories={cat})
                    category_vectors[cat] = cat_fv
                    parts_in_order.append(cat_fv)
                except ValueError:
                    pass
            feature_vectors = np.concatenate(parts_in_order, axis=1)
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            timings["process"].append(time.time() - t0)

            timings["total_gpu"].append(time.time() - t_gpu_start)

            # Compute labels
            labels = compute_all_labels(dataset)

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
            logger.warning(f"Dataset {idx} GPU processing failed: {e}")
            n_skipped += 1
            torch.cuda.empty_cache()
            continue

        if n_saved % 50 == 0 and n_saved > 0:
            elapsed = time.time() - t_total
            rate = n_saved / elapsed * 3600
            avg_gen = np.mean(timings["gen"][-50:]) if timings["gen"] else 0
            avg_gpu = np.mean(timings["total_gpu"][-50:]) if timings["total_gpu"] else 0
            logger.info(
                f"[{n_saved}/{n_datasets}] saved, {n_skipped} skipped | "
                f"{elapsed/60:.1f}min elapsed | {rate:.0f} datasets/hr | "
                f"avg_gen={avg_gen:.1f}s avg_gpu={avg_gpu:.1f}s | "
                f"Last: n_feat={n_feat}, n_samp={n_samp}, fv={feature_vectors.shape}"
            )

    prefetch_thread.join(timeout=5)

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
            f"save={np.mean(timings['save']):.2f}s "
            f"total_gpu={np.mean(timings['total_gpu']):.2f}s"
        )

    manifest = {
        "n_saved": n_saved,
        "n_skipped": n_skipped,
        "seed": seed,
        "min_features": MIN_FEATURES,
        "max_features": MAX_FEATURES,
        "feature_distribution": "uniform",
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "elapsed_min": round(elapsed / 60, 1),
        "generator": "tabpfn_prior (zzhang-cn/tabpfn-synthetic-data)",
        "signal_processor": "enriched=False, extract_gradients=input_only",
        "optimizations": "pipelining + mixed_precision(bfloat16) + flash_attention(SDPA)",
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interpretation data v4 (pipelined + mixed precision)"
    )
    parser.add_argument("--n_datasets", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    if args.device:
        DEVICE = args.device
    cache = Path(args.cache_dir) if args.cache_dir else CACHE_DIR

    generate_and_cache(
        n_datasets=args.n_datasets,
        seed=args.seed,
        start_idx=args.start_idx,
        cache_dir=cache,
        force=args.force,
        use_amp=not args.no_amp,
    )
