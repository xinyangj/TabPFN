#!/usr/bin/env python
"""Generate interpretation training data v5 — 3-stage async pipeline.

Improvements over v4:
  - 3-stage pipeline: CPU gen → GPU fit+extract → CPU process → save
  - GPU and CPU processing run concurrently in separate threads
  - Proper CPU transfer before process() (7x faster processing)
  - Mixed precision (bfloat16) with SDPA/flash attention

Expected throughput: ~1.5-2s/dataset (vs ~3s in v4)

Usage:
    # Single GPU
    python scripts/generate_interpretation_data_v5.py --n_datasets 1000 --device cuda:0

    # Multi-GPU (separate terminals)
    python scripts/generate_interpretation_data_v5.py --n_datasets 1000 --start_idx 0    --seed 100 --device cuda:0
    python scripts/generate_interpretation_data_v5.py --n_datasets 1000 --start_idx 1000 --seed 200 --device cuda:1
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
    format="%(asctime)s %(levelname)s [%(threadName)s] %(message)s",
)
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/interpretation_cache_v5")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MIN_FEATURES = 10
MAX_FEATURES = 150
MAX_TRAIN_SAMPLES = 1000

GEN_QUEUE_SIZE = 2
SIG_QUEUE_SIZE = 2
OUT_QUEUE_SIZE = 4

CATEGORIES = [
    "between_features_attention",
    "between_items_attention",
    "embeddings",
    "mlp_activations",
    "gradients",
]


def _sample_n_features(rng: np.random.Generator) -> int:
    return int(rng.integers(MIN_FEATURES, MAX_FEATURES + 1))


def _sample_n_samples(rng: np.random.Generator) -> int:
    return int(rng.integers(50, MAX_TRAIN_SAMPLES + 1))


def save_dataset(path: Path, record: dict) -> None:
    """Save a single dataset record to .npz + .json."""
    arrays: dict[str, np.ndarray] = {"feature_vectors": record["feature_vectors"]}
    for lm, arr in record["labels"].items():
        arrays[f"label_{lm}"] = arr
    for cat_name, cat_fv in record.get("category_vectors", {}).items():
        arrays[f"cat_{cat_name}"] = cat_fv
    for raw_name, raw_arr in record.get("raw_data", {}).items():
        arrays[f"raw_{raw_name}"] = raw_arr

    np.savez_compressed(path, **arrays)

    meta_path = path.with_suffix(".json")
    safe_meta = {}
    for k, v in record["metadata"].items():
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


# ── Stage 1: CPU data generation ──────────────────────────────────────

def gen_worker(
    gen,
    rng: np.random.Generator,
    gen_queue: queue.Queue,
    index_iter,
    cache_dir: Path,
    force: bool,
):
    """Thread 1: Generate synthetic datasets on CPU."""
    for i in index_iter:
        out_path = cache_dir / f"dataset_{i:06d}.npz"
        n_feat = _sample_n_features(rng)
        n_samp = _sample_n_samples(rng)

        if out_path.exists() and not force:
            gen_queue.put(("skip", i, out_path, None, None, None))
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

            gen_queue.put((
                "data", i, out_path, dataset,
                (X_train, y_train, X_test),
                {"gen_time": gen_time, "n_feat": n_feat, "n_samp": n_samp},
            ))
        except Exception as e:
            logger.warning(f"Dataset {i} generation failed: {e}")
            gen_queue.put(("error", i, out_path, None, None, None))

    gen_queue.put(("done", -1, None, None, None, None))


# ── Stage 2: GPU fit + extract + GPU stats ─────────────────────────

def gpu_worker(
    gen_queue: queue.Queue,
    sig_queue: queue.Queue,
    device: str,
    use_amp: bool,
):
    """Thread 2: GPU TabPFN fit + signal extraction + GPU stats computation.

    Instead of transferring ~1 GB of raw tensors to CPU, computes per-block
    statistics on GPU and transfers only ~20 KB of results.
    """
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.gpu_stats_computer import GPUStatsComputer

    extractor = SignalExtractor(extract_gradients="input_only")
    gpu_stats_computer = GPUStatsComputer(enriched=False)
    amp_enabled = use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    while True:
        item = gen_queue.get()
        msg_type, idx, out_path, dataset, data_tuple, meta = item

        if msg_type == "done":
            sig_queue.put(("done", -1, None, None, None, None, None))
            break
        elif msg_type in ("skip", "error"):
            sig_queue.put((msg_type, idx, out_path, None, None, None, meta))
            continue

        X_train, y_train, X_test = data_tuple

        try:
            t_fit = time.time()
            reg = TabPFNRegressor(n_estimators=1, device=device)
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    reg.fit(X_train, y_train)
            else:
                reg.fit(X_train, y_train)
            fit_time = time.time() - t_fit

            t_ext = time.time()
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    signals = extractor.extract(reg, X_train, y_train, X_test)
            else:
                signals = extractor.extract(reg, X_train, y_train, X_test)
            extract_time = time.time() - t_ext

            n_features = signals["n_features"]

            # Compute stats on GPU and transfer tiny results (~20 KB vs ~1 GB)
            t_stats = time.time()
            gpu_stats = gpu_stats_computer.compute(signals)
            stats_time = time.time() - t_stats

            del reg, signals
            torch.cuda.empty_cache()

            timing_meta = {
                **(meta or {}),
                "fit_time": fit_time,
                "extract_time": extract_time,
                "stats_time": stats_time,
            }

            sig_queue.put(("data", idx, out_path, dataset,
                           (gpu_stats, n_features), data_tuple, timing_meta))

        except Exception as e:
            logger.warning(f"Dataset {idx} GPU failed: {e}")
            torch.cuda.empty_cache()
            sig_queue.put(("error", idx, out_path, None, None, None, meta))


# ── Stage 3: CPU signal processing ────────────────────────────────────

def process_worker(
    sig_queue: queue.Queue,
    out_queue: queue.Queue,
):
    """Thread 3: CPU signal processing from pre-computed GPU stats."""
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
    from tabpfn.interpretation.synthetic_data.label_generator import compute_all_labels

    processor = SignalProcessor(enriched=False)

    while True:
        item = sig_queue.get()
        msg_type, idx, out_path, dataset, stats_tuple, data_tuple, meta = item

        if msg_type == "done":
            out_queue.put(("done", -1, None, None))
            break
        elif msg_type in ("skip", "error"):
            out_queue.put((msg_type, idx, out_path, meta))
            continue

        try:
            gpu_stats, n_features = stats_tuple

            t_proc = time.time()
            category_vectors = {}
            parts = []
            for cat in CATEGORIES:
                try:
                    cat_fv = processor.process_from_stats(
                        gpu_stats, n_features, signal_categories={cat}
                    )
                    category_vectors[cat] = cat_fv
                    parts.append(cat_fv)
                except ValueError:
                    pass

            feature_vectors = np.concatenate(parts, axis=1)
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)
            proc_time = time.time() - t_proc

            labels = compute_all_labels(dataset)

            if labels["binary_direct"].sum() == 0 and labels["binary_ancestry"].sum() == 0:
                out_queue.put(("skip_empty", idx, out_path, meta))
                continue

            X_train, y_train, X_test = data_tuple
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

            timing_meta = {**(meta or {}), "proc_time": proc_time}
            out_queue.put(("data", idx, out_path, (record, timing_meta)))

        except Exception as e:
            logger.warning(f"Dataset {idx} processing failed: {e}")
            out_queue.put(("error", idx, out_path, meta))


# ── Main: save + bookkeeping ──────────────────────────────────────────

def generate_and_cache(
    n_datasets: int = 1000,
    seed: int = 100,
    start_idx: int = 0,
    cache_dir: Path | None = None,
    *,
    force: bool = False,
    use_amp: bool = True,
) -> None:
    """Run 3-stage async pipeline: gen → GPU → process → save."""
    from tabpfn.interpretation.synthetic_data.tabpfn_prior_adapter import (
        TabPFNPriorAdapter,
    )

    if cache_dir is None:
        cache_dir = CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    gen = TabPFNPriorAdapter(seed=seed)
    rng = np.random.default_rng(seed)

    # Advance RNG to start_idx
    for _ in range(start_idx):
        _sample_n_features(rng)
        _sample_n_samples(rng)

    # Create queues
    gen_queue: queue.Queue = queue.Queue(maxsize=GEN_QUEUE_SIZE)
    sig_queue: queue.Queue = queue.Queue(maxsize=SIG_QUEUE_SIZE)
    out_queue: queue.Queue = queue.Queue(maxsize=OUT_QUEUE_SIZE)

    # Start pipeline threads
    index_iter = range(start_idx, start_idx + n_datasets)

    t1 = threading.Thread(
        target=gen_worker,
        args=(gen, rng, gen_queue, index_iter, cache_dir, force),
        name="GenWorker",
        daemon=True,
    )
    t2 = threading.Thread(
        target=gpu_worker,
        args=(gen_queue, sig_queue, DEVICE, use_amp),
        name="GPUWorker",
        daemon=True,
    )
    t3 = threading.Thread(
        target=process_worker,
        args=(sig_queue, out_queue),
        name="ProcessWorker",
        daemon=True,
    )

    t1.start()
    t2.start()
    t3.start()

    amp_str = "bfloat16" if use_amp else "disabled"
    logger.info(
        f"Pipeline started: {n_datasets} datasets, device={DEVICE}, amp={amp_str}, "
        f"start_idx={start_idx}, seed={seed}"
    )

    n_saved = 0
    n_skipped = 0
    t_total = time.time()
    timings = {"gen": [], "fit": [], "extract": [], "stats": [], "proc": [], "save": [], "total": []}

    while True:
        item = out_queue.get()
        msg_type, idx, out_path = item[0], item[1], item[2]

        if msg_type == "done":
            break
        elif msg_type in ("skip", "skip_empty"):
            if msg_type == "skip":
                n_saved += 1  # Already existed on disk
            else:
                n_skipped += 1
            continue
        elif msg_type == "error":
            n_skipped += 1
            continue

        record, timing_meta = item[3]

        # Save
        t_save = time.time()
        save_dataset(out_path, record)
        save_time = time.time() - t_save

        n_saved += 1

        # Collect timings
        for tk in ("gen_time", "fit_time", "extract_time", "stats_time", "proc_time"):
            short = tk.replace("_time", "")
            if tk in timing_meta:
                timings[short].append(timing_meta[tk])
        timings["save"].append(save_time)
        total_t = sum(timing_meta.get(k, 0) for k in
                      ("gen_time", "fit_time", "extract_time", "stats_time", "proc_time"))
        timings["total"].append(total_t + save_time)

        if n_saved % 25 == 0 and n_saved > 0:
            elapsed = time.time() - t_total
            rate = n_saved / elapsed * 3600
            fv_shape = record["feature_vectors"].shape

            last_n = 25
            avg = lambda k: np.mean(timings[k][-last_n:]) if timings[k] else 0
            logger.info(
                f"[{n_saved}/{n_datasets}] {n_skipped} skip | "
                f"{elapsed/60:.1f}min | {rate:.0f}/hr | "
                f"gen={avg('gen'):.2f} fit={avg('fit'):.2f} ext={avg('extract'):.2f} "
                f"stats={avg('stats'):.3f} proc={avg('proc'):.3f} save={avg('save'):.2f} | "
                f"fv={fv_shape} | queues: gen={gen_queue.qsize()} sig={sig_queue.qsize()} out={out_queue.qsize()}"
            )

    # Wait for threads to finish
    t1.join(timeout=10)
    t2.join(timeout=10)
    t3.join(timeout=10)

    elapsed = time.time() - t_total
    logger.info(f"Done: {n_saved} saved, {n_skipped} skipped in {elapsed/60:.1f} min")
    if timings["fit"]:
        logger.info(
            f"Avg timings: gen={np.mean(timings['gen']):.2f}s "
            f"fit={np.mean(timings['fit']):.2f}s "
            f"extract={np.mean(timings['extract']):.2f}s "
            f"stats={np.mean(timings['stats']):.3f}s "
            f"proc={np.mean(timings['proc']):.3f}s "
            f"save={np.mean(timings['save']):.2f}s "
            f"| total_per_dataset={np.mean(timings['total']):.2f}s "
            f"| wall_rate={n_saved/elapsed*3600:.0f}/hr"
        )

    manifest = {
        "n_saved": n_saved,
        "n_skipped": n_skipped,
        "seed": seed,
        "start_idx": start_idx,
        "min_features": MIN_FEATURES,
        "max_features": MAX_FEATURES,
        "feature_distribution": "uniform",
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "elapsed_min": round(elapsed / 60, 1),
        "wall_rate_per_hr": round(n_saved / max(elapsed, 1) * 3600, 0),
        "categories": CATEGORIES,
        "generator": "tabpfn_prior (zzhang-cn/tabpfn-synthetic-data)",
        "signal_processor": "enriched=False, extract_gradients=input_only",
        "optimizations": "3-stage async pipeline + GPU-side stats + mixed_precision(bfloat16) + SDPA",
    }
    with open(cache_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {cache_dir / 'manifest.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate interpretation data v5 (3-stage async pipeline)"
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
