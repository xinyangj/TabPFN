#!/usr/bin/env python
"""Enrich cached interpretation data — re-run TabPFN inference with enriched=True.

Reads raw_X_train, raw_y_train, raw_X_test from existing v6_1m .npz files,
re-runs TabPFN inference to extract enriched signals (3952d), and saves to
a new output directory.

Usage:
    # Single GPU
    python scripts/enrich_cached_data.py --device cuda:0

    # Multi-GPU (separate terminals)
    python scripts/enrich_cached_data.py --device cuda:0 --start_idx 0 --end_idx 476090
    python scripts/enrich_cached_data.py --device cuda:1 --start_idx 476090 --end_idx 952180
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import shutil
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

INPUT_DIR = Path("data/interpretation_cache_v6_1m")
OUTPUT_DIR = Path("data/interpretation_cache_v1_1m_enriched")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

ENRICHED_CATEGORIES = [
    "between_features_attention",
    "between_items_attention",
    "embeddings",
    "mlp_activations",
    "gradients",
    "value_contributions",
]

READ_QUEUE_SIZE = 4
SIG_QUEUE_SIZE = 2
OUT_QUEUE_SIZE = 4


# ── Stage 1: Read cached raw data ────────────────────────────────────

def read_worker(
    file_list: list[tuple[int, Path, Path]],
    read_queue: queue.Queue,
    output_dir: Path,
    force: bool,
):
    """Thread 1: Load raw data from source .npz files."""
    for idx, src_npz, out_npz in file_list:
        if out_npz.exists() and not force:
            read_queue.put(("skip", idx, out_npz, None, None, None))
            continue

        try:
            t0 = time.time()
            data = np.load(src_npz, allow_pickle=True)

            X_train = data["raw_X_train"].astype(np.float32)
            y_train = data["raw_y_train"].astype(np.float32)
            X_test = data["raw_X_test"].astype(np.float32)

            # Copy labels
            labels = {}
            for k in data.files:
                if k.startswith("label_"):
                    labels[k] = data[k]

            read_time = time.time() - t0

            read_queue.put((
                "data", idx, out_npz, src_npz,
                (X_train, y_train, X_test, labels),
                {"read_time": read_time},
            ))
        except Exception as e:
            logger.warning(f"Dataset {idx} read failed: {e}")
            read_queue.put(("error", idx, out_npz, None, None, None))

    read_queue.put(("done", -1, None, None, None, None))


# ── Stage 2: GPU fit + extract + GPU stats ────────────────────────────

def gpu_worker(
    read_queue: queue.Queue,
    sig_queue: queue.Queue,
    device: str,
    use_amp: bool,
):
    """Thread 2: GPU TabPFN fit + enriched signal extraction + GPU stats."""
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.gpu_stats_computer import GPUStatsComputer

    extractor = SignalExtractor(extract_gradients=True)
    gpu_stats_computer = GPUStatsComputer(enriched=True)
    amp_enabled = use_amp and torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    while True:
        item = read_queue.get()
        msg_type, idx, out_npz, src_npz, data_tuple, meta = item

        if msg_type == "done":
            sig_queue.put(("done", -1, None, None, None, None, None))
            break
        elif msg_type in ("skip", "error"):
            sig_queue.put((msg_type, idx, out_npz, src_npz, None, None, meta))
            continue

        X_train, y_train, X_test, labels = data_tuple

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

            sig_queue.put(("data", idx, out_npz, src_npz,
                           (gpu_stats, n_features, labels),
                           (X_train, y_train, X_test),
                           timing_meta))

        except Exception as e:
            logger.warning(f"Dataset {idx} GPU failed: {e}")
            torch.cuda.empty_cache()
            sig_queue.put(("error", idx, out_npz, src_npz, None, None, meta))


# ── Stage 3: CPU signal processing + save ─────────────────────────────

def process_worker(
    sig_queue: queue.Queue,
    out_queue: queue.Queue,
):
    """Thread 3: CPU signal processing from pre-computed GPU stats."""
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

    processor = SignalProcessor(enriched=True)

    while True:
        item = sig_queue.get()
        msg_type, idx, out_npz, src_npz, stats_tuple, raw_data, meta = item

        if msg_type == "done":
            out_queue.put(("done", -1, None, None))
            break
        elif msg_type in ("skip", "error"):
            out_queue.put((msg_type, idx, out_npz, meta))
            continue

        try:
            gpu_stats, n_features, labels = stats_tuple
            X_train, y_train, X_test = raw_data

            t_proc = time.time()
            category_vectors = {}
            parts = []
            for cat in ENRICHED_CATEGORIES:
                try:
                    cat_fv = processor.process_from_stats(
                        gpu_stats, n_features, signal_categories={cat}
                    )
                    category_vectors[cat] = cat_fv
                    parts.append(cat_fv)
                except ValueError:
                    pass

            feature_vectors = np.concatenate(parts, axis=1)
            feature_vectors = np.nan_to_num(
                feature_vectors, nan=0.0, posinf=0.0, neginf=0.0
            )
            proc_time = time.time() - t_proc

            # Build record
            arrays = {"feature_vectors": feature_vectors}
            for lbl_name, lbl_arr in labels.items():
                arrays[lbl_name] = lbl_arr
            for cat_name, cat_fv in category_vectors.items():
                arrays[f"cat_{cat_name}"] = cat_fv
            arrays["raw_X_train"] = X_train
            arrays["raw_y_train"] = y_train
            arrays["raw_X_test"] = X_test

            timing_meta = {**(meta or {}), "proc_time": proc_time}
            out_queue.put(("data", idx, out_npz,
                           (arrays, src_npz, timing_meta)))

        except Exception as e:
            logger.warning(f"Dataset {idx} processing failed: {e}")
            out_queue.put(("error", idx, out_npz, meta))


# ── Main: save + bookkeeping ──────────────────────────────────────────

def enrich_cache(
    input_dir: Path,
    output_dir: Path,
    start_idx: int = 0,
    end_idx: int | None = None,
    *,
    force: bool = False,
    use_amp: bool = True,
) -> None:
    """Run 3-stage async pipeline: read → GPU → process → save."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover source .npz files
    all_npz = sorted(input_dir.glob("dataset_*.npz"))
    if not all_npz:
        logger.error(f"No dataset_*.npz found in {input_dir}")
        return

    # Parse indices and filter by range
    file_list = []
    for p in all_npz:
        idx = int(p.stem.replace("dataset_", ""))
        if idx < start_idx:
            continue
        if end_idx is not None and idx >= end_idx:
            continue
        out_npz = output_dir / p.name
        file_list.append((idx, p, out_npz))

    n_total = len(file_list)
    if n_total == 0:
        logger.info("No datasets in specified range")
        return

    logger.info(f"Found {n_total} datasets to process (idx {start_idx}-{end_idx or 'end'})")

    # Create queues
    read_queue: queue.Queue = queue.Queue(maxsize=READ_QUEUE_SIZE)
    sig_queue: queue.Queue = queue.Queue(maxsize=SIG_QUEUE_SIZE)
    out_queue: queue.Queue = queue.Queue(maxsize=OUT_QUEUE_SIZE)

    # Start pipeline threads
    t1 = threading.Thread(
        target=read_worker,
        args=(file_list, read_queue, output_dir, force),
        name="ReadWorker",
        daemon=True,
    )
    t2 = threading.Thread(
        target=gpu_worker,
        args=(read_queue, sig_queue, DEVICE, use_amp),
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
        f"Pipeline started: {n_total} datasets, device={DEVICE}, amp={amp_str}"
    )

    n_saved = 0
    n_skipped = 0
    n_errors = 0
    t_total = time.time()
    timings = {
        "read": [], "fit": [], "extract": [], "stats": [],
        "proc": [], "save": [], "total": [],
    }

    while True:
        item = out_queue.get()
        msg_type, idx, out_npz = item[0], item[1], item[2]

        if msg_type == "done":
            break
        elif msg_type == "skip":
            n_skipped += 1
            continue
        elif msg_type == "error":
            n_errors += 1
            continue

        arrays, src_npz, timing_meta = item[3]

        # Save enriched .npz
        t_save = time.time()
        np.savez_compressed(out_npz, **arrays)
        save_time = time.time() - t_save

        # Copy JSON metadata from source
        src_json = Path(src_npz).with_suffix(".json")
        out_json = out_npz.with_suffix(".json")
        if src_json.exists() and not out_json.exists():
            shutil.copy2(src_json, out_json)

        n_saved += 1

        # Collect timings
        for tk in ("read_time", "fit_time", "extract_time", "stats_time", "proc_time"):
            short = tk.replace("_time", "")
            if tk in timing_meta:
                timings[short].append(timing_meta[tk])
        timings["save"].append(save_time)
        total_t = sum(timing_meta.get(k, 0) for k in
                      ("read_time", "fit_time", "extract_time", "stats_time", "proc_time"))
        timings["total"].append(total_t + save_time)

        if n_saved % 25 == 0 and n_saved > 0:
            elapsed = time.time() - t_total
            processed = n_saved + n_skipped + n_errors
            rate = n_saved / elapsed * 3600
            fv_shape = arrays["feature_vectors"].shape

            last_n = 25
            avg = lambda k: np.mean(timings[k][-last_n:]) if timings[k] else 0
            logger.info(
                f"[{processed}/{n_total}] saved={n_saved} skip={n_skipped} err={n_errors} | "
                f"{elapsed/60:.1f}min | {rate:.0f}/hr | "
                f"read={avg('read'):.3f} fit={avg('fit'):.2f} ext={avg('extract'):.2f} "
                f"stats={avg('stats'):.3f} proc={avg('proc'):.3f} save={avg('save'):.2f} | "
                f"fv={fv_shape}"
            )

    # Wait for threads to finish
    t1.join(timeout=10)
    t2.join(timeout=10)
    t3.join(timeout=10)

    elapsed = time.time() - t_total
    logger.info(
        f"Done: {n_saved} saved, {n_skipped} skipped, {n_errors} errors "
        f"in {elapsed/60:.1f} min"
    )
    if timings["fit"]:
        logger.info(
            f"Avg timings: read={np.mean(timings['read']):.3f}s "
            f"fit={np.mean(timings['fit']):.2f}s "
            f"extract={np.mean(timings['extract']):.2f}s "
            f"stats={np.mean(timings['stats']):.3f}s "
            f"proc={np.mean(timings['proc']):.3f}s "
            f"save={np.mean(timings['save']):.2f}s "
            f"| total_per_dataset={np.mean(timings['total']):.2f}s "
            f"| wall_rate={n_saved/max(elapsed,1)*3600:.0f}/hr"
        )

    # Write manifest
    manifest = {
        "n_saved": n_saved,
        "n_skipped": n_skipped,
        "n_errors": n_errors,
        "source_dir": str(input_dir),
        "start_idx": start_idx,
        "end_idx": end_idx,
        "categories": ENRICHED_CATEGORIES,
        "signal_processor": "enriched=True, extract_gradients=True",
        "feature_dims": int(arrays["feature_vectors"].shape[1]) if n_saved > 0 else 0,
        "elapsed_min": round(elapsed / 60, 1),
        "wall_rate_per_hr": round(n_saved / max(elapsed, 1) * 3600, 0),
        "avg_timings": {
            k: round(float(np.mean(v)), 4) if v else 0
            for k, v in timings.items()
        },
    }
    manifest_path = output_dir / f"manifest_{DEVICE.replace(':', '_')}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enrich cached interpretation data with full signals (3952d)"
    )
    parser.add_argument("--input_dir", type=str, default=str(INPUT_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    args = parser.parse_args()

    if args.device:
        DEVICE = args.device

    enrich_cache(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        force=args.force,
        use_amp=not args.no_amp,
    )
