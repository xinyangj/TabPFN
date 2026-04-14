#!/usr/bin/env python3
"""Validate GNW pipeline alignment with DREAM4-10 — 5K trials with corrected row alignment.

For each of the 5 DREAM4-10 networks:
  1. Load signed gold standard (exact topology)
  2. Load actual expression data (WT, KO, KD, MF, TS)
  3. Run GNW transform+simulate N times (different random kinetic params)
  4. Select the run whose KO+KD output best matches actual DREAM4-10 (with correct row alignment)
  5. Compare ALL data types with comprehensive metrics
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import tempfile
import glob as globmod
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
GNW_JAR = REPO_ROOT / "data" / "gnw" / "gnw-3.1.2b.jar"
GNW_SETTINGS = REPO_ROOT / "data" / "gnw" / "settings_dream4_size10.txt"

DREAM4_COMPLETE = REPO_ROOT / "data" / "dream4" / "dream4"
DREAM4_SUPP = Path("/tmp/DREAM4 in-silico challenge/Size 10/Supplementary information")
DREAM4_SIGNED_GS = REPO_ROOT / "data" / "dream4" / "dream4"

GENE_NAMES = [f"G{i}" for i in range(1, 11)]


def run_gnw(args: list[str], cwd: str = "/tmp", timeout: int = 120) -> str:
    cmd = ["java", "-jar", str(GNW_JAR)] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def parse_steady_state(path: str) -> tuple[list[str], np.ndarray]:
    names, rows = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') or (not line[0].isdigit() and line[0] != '-'):
                names = [n.strip('"') for n in line.split("\t")]
                continue
            try:
                rows.append([float(x) for x in line.split("\t")])
            except ValueError:
                continue
    return names, np.array(rows, dtype=np.float64) if rows else np.empty((0, 0))


def parse_timeseries(path: str) -> tuple[list[str], np.ndarray]:
    names, rows = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') or line.startswith("Time") or line.startswith("'"):
                parts = [n.strip('"') for n in line.split("\t")]
                names = parts[1:]
                continue
            try:
                vals = [float(x) for x in line.split("\t")]
                rows.append(vals[1:])
            except ValueError:
                continue
    return names, np.array(rows, dtype=np.float64) if rows else np.empty((0, 0))


def reorder_cols(names: list[str], data: np.ndarray) -> np.ndarray:
    """Reorder columns to standard G1..G10 order."""
    if data.size == 0:
        return data
    idx = [names.index(g) for g in GENE_NAMES]
    return data[:, idx]


def reorder_rows_and_cols(names: list[str], data: np.ndarray) -> np.ndarray:
    """Reorder BOTH rows and columns for KO/KD (row i = perturbation of gene i)."""
    if data.size == 0:
        return data
    idx = [names.index(g) for g in GENE_NAMES]
    return data[idx][:, idx]


def load_dream4_network(net_idx: int) -> dict:
    base = DREAM4_COMPLETE / f"insilico_size10_{net_idx}"
    result = {}
    for dtype, suffix, parser in [
        ("wt", "wildtype", parse_steady_state),
        ("ko", "knockouts", parse_steady_state),
        ("kd", "knockdowns", parse_steady_state),
        ("mf", "multifactorial", parse_steady_state),
        ("ts", "timeseries", parse_timeseries),
    ]:
        names, data = parser(str(base / f"insilico_size10_{net_idx}_{suffix}.tsv"))
        if names and data.size > 0:
            result[dtype] = reorder_cols(names, data)
    return result


def load_signed_goldstandard(net_idx: int) -> str:
    """Load signed gold standard — try supplementary archive first, then reconstructed."""
    # Try original supplementary
    path = DREAM4_SUPP / f"insilico_size10_{net_idx}" / "Goldstandard" / f"insilico_size10_{net_idx}_goldstandard_signed.tsv"
    if path.exists():
        return path.read_text()
    # Fall back to reconstructed from KO data
    path2 = DREAM4_SIGNED_GS / f"net{net_idx}_signed.tsv"
    if path2.exists():
        return path2.read_text()
    raise FileNotFoundError(f"No signed gold standard found for net {net_idx}")


def clean_gnw_output(name: str):
    for f in globmod.glob(f"/tmp/{name}_*"):
        os.remove(f)
    for f in [f"/tmp/{name}.xml", f"/tmp/{name}_goldstandard.tsv", f"/tmp/{name}_goldstandard_signed.tsv"]:
        if os.path.exists(f):
            os.remove(f)


def run_single_simulation(gold_standard_tsv: str, net_name: str) -> dict | None:
    """Run one GNW transform+simulate. Output goes to /tmp/ (cwd)."""
    clean_gnw_output(net_name)

    gs_path = f"/tmp/{net_name}.tsv"
    with open(gs_path, "w") as f:
        f.write(gold_standard_tsv)

    try:
        run_gnw(["--transform", "-c", str(GNW_SETTINGS), "--input-net", gs_path,
                 "--output-net-format=4", "--output-path", "/tmp"], cwd="/tmp", timeout=30)
    except subprocess.TimeoutExpired:
        return None

    sbml_path = f"/tmp/{net_name}.xml"
    if not os.path.exists(sbml_path):
        return None

    try:
        run_gnw(["--simulate", "-c", str(GNW_SETTINGS), "--input-net", sbml_path,
                 "--output-path", "/tmp"], cwd="/tmp", timeout=120)
    except subprocess.TimeoutExpired:
        return None

    result = {}
    for dtype, suffix, parser in [
        ("wt", "_wildtype.tsv", parse_steady_state),
        ("ko", "_knockouts.tsv", parse_steady_state),
        ("kd", "_knockdowns.tsv", parse_steady_state),
        ("mf", "_multifactorial.tsv", parse_steady_state),
        ("ts", "_dream4_timeseries.tsv", parse_timeseries),
    ]:
        fpath = f"/tmp/{net_name}{suffix}"
        if os.path.exists(fpath):
            names, data = parser(fpath)
            if names and data.size > 0:
                result[f"{dtype}_names"] = names
                # KO/KD: reorder both rows and cols; others: cols only
                if dtype in ("ko", "kd"):
                    result[dtype] = reorder_rows_and_cols(names, data)
                else:
                    result[dtype] = reorder_cols(names, data)

    if "ko" not in result or "kd" not in result:
        return None

    # Save SBML content
    if os.path.exists(sbml_path):
        result["sbml_content"] = open(sbml_path).read()

    return result


def compute_calibration_mse(gen: dict, act: dict) -> float:
    """MSE on KO+KD with correct row alignment (already reordered)."""
    mse_total, n_total = 0.0, 0
    for dtype in ["ko", "kd"]:
        g, a = gen[dtype], act[dtype]
        if g.shape != a.shape:
            return float("inf")
        for r in range(g.shape[0]):
            for c in range(g.shape[1]):
                if dtype == "ko" and r == c:
                    continue  # skip knocked-out gene
                diff = g[r, c] - a[r, c]
                mse_total += diff * diff
                n_total += 1
    return mse_total / n_total if n_total > 0 else float("inf")


def compute_detailed_metrics(gen: np.ndarray, act: np.ndarray) -> dict:
    """Compute comprehensive alignment metrics between two matrices."""
    m = {"shape": gen.shape}
    flat_g, flat_a = gen.flatten(), act.flatten()

    # Element-wise
    m["elem_pearson"], _ = stats.pearsonr(flat_g, flat_a)
    m["elem_spearman"], _ = stats.spearmanr(flat_g, flat_a)
    m["mse"] = float(np.mean((gen - act) ** 2))
    m["mae"] = float(np.mean(np.abs(gen - act)))

    # KS
    m["ks_stat"], m["ks_p"] = stats.ks_2samp(flat_g, flat_a)

    # Gene mean profile
    m["profile_r"], _ = stats.pearsonr(gen.mean(0), act.mean(0))

    n_samples = gen.shape[0]

    # Per-gene correlation
    if n_samples > 2:
        grs = []
        for g in range(gen.shape[1]):
            if np.std(act[:, g]) > 1e-10 and np.std(gen[:, g]) > 1e-10:
                r, _ = stats.pearsonr(act[:, g], gen[:, g])
                grs.append(r)
        if grs:
            m["per_gene_r_mean"] = float(np.mean(grs))
            m["per_gene_r_std"] = float(np.std(grs))

    # Per-sample correlation
    srs = []
    for s in range(n_samples):
        if np.std(act[s]) > 1e-10 and np.std(gen[s]) > 1e-10:
            r, _ = stats.pearsonr(act[s], gen[s])
            srs.append(r)
    if srs:
        m["per_sample_r_mean"] = float(np.mean(srs))
        m["per_sample_r_std"] = float(np.std(srs))

    # Gene-gene correlation structure
    if n_samples > 2:
        ac = np.corrcoef(act.T)
        gc = np.corrcoef(gen.T)
        mask = np.triu_indices(gen.shape[1], k=1)
        m["gene_gene_r"], _ = stats.pearsonr(ac[mask], gc[mask])
        m["gene_gene_frob"] = float(np.linalg.norm(ac - gc, "fro"))

    return m


def calibrate_network(net_idx: int, n_trials: int, actual: dict) -> tuple[dict | None, float]:
    """Run n_trials GNW simulations and return best matching one."""
    gs_text = load_signed_goldstandard(net_idx)
    net_name = f"CalNet{net_idx}"

    best_result, best_mse = None, float("inf")
    n_ok = 0

    for trial in range(n_trials):
        res = run_single_simulation(gs_text, net_name)
        if res is None:
            continue
        n_ok += 1

        mse = compute_calibration_mse(res, actual)
        if mse < best_mse:
            best_mse = mse
            best_result = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in res.items() if not k.endswith("_names")}

        if (trial + 1) % 100 == 0:
            log.info(f"    Trial {trial+1}/{n_trials}: {n_ok} ok, best MSE={best_mse:.6f}")

    clean_gnw_output(net_name)
    log.info(f"  Net {net_idx}: best MSE={best_mse:.6f} ({n_ok}/{n_trials} ok)")
    return best_result, best_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=5000)
    parser.add_argument("--output", type=str, default="results/interpretation_experiments/dream4_alignment_5k.json")
    args = parser.parse_args()

    log.info(f"DREAM4-10 Alignment Validation — {args.n_trials} trials/net")

    # Load actual data
    actual_data = {}
    for n in range(1, 6):
        actual_data[n] = load_dream4_network(n)
        log.info(f"  Net {n}: KO={actual_data[n]['ko'].shape}, KD={actual_data[n]['kd'].shape}")

    # Calibrate
    all_results = {}
    all_metrics = {}

    for net_idx in range(1, 6):
        log.info(f"\n{'='*60}")
        log.info(f"Network {net_idx}: {args.n_trials} trials")
        log.info(f"{'='*60}")

        best, mse = calibrate_network(net_idx, args.n_trials, actual_data[net_idx])
        if best is None:
            log.warning(f"  Net {net_idx}: FAILED")
            continue

        all_results[net_idx] = best

        # Compute detailed metrics
        net_metrics = {"calibration_mse": mse}
        for dtype, label, role in [
            ("wt", "Wild-type", "held_out"),
            ("ko", "Knockouts", "target"),
            ("kd", "Knockdowns", "target"),
            ("mf", "Multifactorial", "held_out"),
            ("ts", "Timeseries", "held_out"),
        ]:
            if dtype in best and dtype in actual_data[net_idx]:
                m = compute_detailed_metrics(best[dtype], actual_data[net_idx][dtype])
                m["role"] = role
                net_metrics[dtype] = m

        all_metrics[net_idx] = net_metrics

    # Print results
    dtypes = ["wt", "ko", "kd", "mf", "ts"]
    labels = ["WT", "KO", "KD", "MF", "TS"]

    for title, key in [
        ("Element-wise Pearson r", "elem_pearson"),
        ("Gene mean-profile r", "profile_r"),
        ("Per-gene r (across samples)", "per_gene_r_mean"),
        ("Per-sample r (across genes)", "per_sample_r_mean"),
        ("Gene-gene corr structure r", "gene_gene_r"),
        ("MSE", "mse"),
        ("KS distance", "ks_stat"),
    ]:
        print(f"\n{'='*90}")
        print(f"TABLE: {title}")
        print(f"{'='*90}")
        use_dts = [d for d in dtypes if any(
            key in all_metrics.get(n, {}).get(d, {}) for n in range(1, 6))]
        use_lbs = [l for d, l in zip(dtypes, labels) if d in use_dts]
        fmt = ".4f" if key == "mse" else ".4f" if key == "ks_stat" else ".3f"
        print(f"{'Net':>4}", "".join(f"{l:>8}" for l in use_lbs), f"{'Mean':>8}")

        avg_vals = {d: [] for d in use_dts}
        for n in range(1, 6):
            if n not in all_metrics:
                continue
            vals = []
            for d in use_dts:
                v = all_metrics[n].get(d, {}).get(key, float("nan"))
                vals.append(v)
                if not np.isnan(v):
                    avg_vals[d].append(v)
            m = np.nanmean(vals)
            print(f"{n:>4}", "".join(f"{v:>8{fmt}}" for v in vals), f"{m:>8{fmt}}")

        avgs = [np.mean(avg_vals[d]) if avg_vals[d] else float("nan") for d in use_dts]
        print(f" AVG", "".join(f"{a:>8{fmt}}" for a in avgs), f"{np.nanmean(avgs):>8{fmt}}")

    # Calibration MSE summary
    print(f"\n{'='*90}")
    print("Calibration MSE per network")
    print(f"{'='*90}")
    for n in range(1, 6):
        if n in all_metrics:
            print(f"  Net {n}: {all_metrics[n]['calibration_mse']:.6f}")

    # Save JSON
    output_path = REPO_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for n, m in all_metrics.items():
        net_s = {"calibration_mse": m["calibration_mse"]}
        for d in dtypes:
            if d in m:
                net_s[d] = {k: float(v) if isinstance(v, (np.floating, float)) else v
                           for k, v in m[d].items() if k != "shape"}
        serializable[str(n)] = net_s
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    log.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
