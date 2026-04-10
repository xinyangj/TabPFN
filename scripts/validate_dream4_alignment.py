#!/usr/bin/env python3
"""Validate GNW pipeline alignment with DREAM4-10 by calibrating kinetic parameters.

For each of the 5 DREAM4-10 networks:
  1. Load signed gold standard (exact topology)
  2. Load actual expression data (WT, KO, KD, MF, TS)
  3. Run GNW transform+simulate N times (different random kinetic params)
  4. Select the run whose KO+KD output best matches actual DREAM4-10
  5. Compare ALL data types from that calibrated run

This validates whether our GNW pipeline can reproduce DREAM4-10 expression
when given the exact same network topology.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
GNW_JAR = REPO_ROOT / "data" / "gnw" / "gnw-3.1.2b.jar"
GNW_SETTINGS = REPO_ROOT / "data" / "gnw" / "settings_dream4_size10.txt"

DREAM4_COMPLETE = Path("/tmp/dream4_complete")
DREAM4_SUPP = Path("/tmp/DREAM4 in-silico challenge/Size 10/Supplementary information")

GENE_NAMES = [f"G{i}" for i in range(1, 11)]


# ---------------------------------------------------------------------------
# GNW helpers
# ---------------------------------------------------------------------------
def run_gnw(args: list[str], cwd: str, timeout: int = 120) -> str:
    cmd = ["java", "-jar", str(GNW_JAR)] + args
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.stdout + result.stderr


def parse_steady_state(path: str) -> tuple[list[str], np.ndarray]:
    """Parse GNW steady-state TSV → (gene_names, (n_samples, n_genes))."""
    names = []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') or (not line[0].isdigit() and line[0] != '-'):
                # Header line
                names = [n.strip('"') for n in line.split("\t")]
                continue
            parts = line.split("\t")
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    arr = np.array(rows, dtype=np.float64) if rows else np.empty((0, 0))
    return names, arr


def parse_timeseries(path: str) -> tuple[list[str], np.ndarray]:
    """Parse GNW time-series TSV → (gene_names, (n_samples, n_genes)) dropping time column."""
    names = []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('"') or line.startswith("Time") or line.startswith("'"):
                parts = [n.strip('"') for n in line.split("\t")]
                names = parts[1:]  # drop "Time"
                continue
            parts = line.split("\t")
            try:
                vals = [float(x) for x in parts]
                rows.append(vals[1:])  # drop time column
            except ValueError:
                continue
    arr = np.array(rows, dtype=np.float64) if rows else np.empty((0, 0))
    return names, arr


def reorder_to_standard(names: list[str], data: np.ndarray) -> np.ndarray:
    """Reorder columns so genes are in G1, G2, ..., G10 order."""
    if data.size == 0:
        return data
    name_to_idx = {n: i for i, n in enumerate(names)}
    order = [name_to_idx[g] for g in GENE_NAMES]
    return data[:, order]


# ---------------------------------------------------------------------------
# Load actual DREAM4-10 data
# ---------------------------------------------------------------------------
def load_dream4_network(net_idx: int) -> dict:
    """Load actual DREAM4-10 data for network net_idx (1-5)."""
    base = DREAM4_COMPLETE / f"insilico_size10_{net_idx}"

    result = {}
    # Wild-type
    names, wt = parse_steady_state(str(base / f"insilico_size10_{net_idx}_wildtype.tsv"))
    result["wt"] = reorder_to_standard(names, wt) if names else wt

    # Knockouts
    names, ko = parse_steady_state(str(base / f"insilico_size10_{net_idx}_knockouts.tsv"))
    result["ko"] = reorder_to_standard(names, ko) if names else ko

    # Knockdowns
    names, kd = parse_steady_state(str(base / f"insilico_size10_{net_idx}_knockdowns.tsv"))
    result["kd"] = reorder_to_standard(names, kd) if names else kd

    # Multifactorial
    names, mf = parse_steady_state(str(base / f"insilico_size10_{net_idx}_multifactorial.tsv"))
    result["mf"] = reorder_to_standard(names, mf) if names else mf

    # Time series
    names, ts = parse_timeseries(str(base / f"insilico_size10_{net_idx}_timeseries.tsv"))
    result["ts"] = reorder_to_standard(names, ts) if names else ts

    return result


def load_signed_goldstandard(net_idx: int) -> str:
    """Load DREAM4-10 signed gold standard as text (GNW TSV format)."""
    path = DREAM4_SUPP / f"insilico_size10_{net_idx}" / "Goldstandard" / f"insilico_size10_{net_idx}_goldstandard_signed.tsv"
    return path.read_text()


# ---------------------------------------------------------------------------
# Single GNW run: transform + simulate
# ---------------------------------------------------------------------------
def run_single_gnw_simulation(gold_standard_tsv: str, net_name: str, tmpdir: str) -> dict | None:
    """Run one GNW transform+simulate cycle. Returns parsed expression data or None."""
    # Write gold standard to temp file
    gs_path = os.path.join(tmpdir, f"{net_name}.tsv")
    with open(gs_path, "w") as f:
        f.write(gold_standard_tsv)

    # Transform: TSV → SBML (assigns random kinetic params)
    try:
        output = run_gnw([
            "--transform",
            "-c", str(GNW_SETTINGS),
            "--input-net", gs_path,
            "--output-net-format=4",
            "--output-path", tmpdir,
            "--network-name", net_name,
        ], cwd=tmpdir, timeout=30)
    except subprocess.TimeoutExpired:
        return None

    sbml_path = os.path.join(tmpdir, f"{net_name}.xml")
    if not os.path.exists(sbml_path):
        return None

    # Simulate
    try:
        output = run_gnw([
            "--simulate",
            "-c", str(GNW_SETTINGS),
            "--input-net", sbml_path,
        ], cwd=tmpdir, timeout=120)
    except subprocess.TimeoutExpired:
        return None

    # Parse all output types
    result = {}
    prefix = os.path.join(tmpdir, net_name)

    for dtype, parser, suffix in [
        ("wt", parse_steady_state, "_wildtype.tsv"),
        ("ko", parse_steady_state, "_knockouts.tsv"),
        ("kd", parse_steady_state, "_knockdowns.tsv"),
        ("mf", parse_steady_state, "_multifactorial.tsv"),
        ("ts", parse_timeseries, "_dream4_timeseries.tsv"),
    ]:
        fpath = prefix + suffix
        if os.path.exists(fpath):
            names, data = parser(fpath)
            if names and data.size > 0:
                result[dtype] = reorder_to_standard(names, data)

    if "ko" not in result or "kd" not in result:
        return None

    result["sbml_path"] = sbml_path
    return result


# ---------------------------------------------------------------------------
# Calibration: find best kinetic parameters
# ---------------------------------------------------------------------------
def compute_ko_kd_mse(generated: dict, actual: dict) -> float:
    """MSE between generated and actual KO+KD expression (ignoring knocked-out gene zeros)."""
    mse_total = 0.0
    n_total = 0

    for dtype in ["ko", "kd"]:
        gen = generated[dtype]
        act = actual[dtype]
        if gen.shape != act.shape:
            return float("inf")

        for row_idx in range(gen.shape[0]):
            for col_idx in range(gen.shape[1]):
                # For KO: skip the knocked-out gene (value = 0)
                if dtype == "ko" and row_idx == col_idx:
                    continue
                diff = gen[row_idx, col_idx] - act[row_idx, col_idx]
                mse_total += diff * diff
                n_total += 1

    return mse_total / n_total if n_total > 0 else float("inf")


def calibrate_network(net_idx: int, n_trials: int = 500) -> tuple[dict | None, float]:
    """Run n_trials GNW simulations and return the best-matching one."""
    gs_text = load_signed_goldstandard(net_idx)
    actual = load_dream4_network(net_idx)

    log.info(f"  Calibrating net {net_idx}: {n_trials} trials, "
             f"actual KO shape={actual['ko'].shape}, KD shape={actual['kd'].shape}")

    best_result = None
    best_mse = float("inf")
    n_success = 0

    for trial in range(n_trials):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_single_gnw_simulation(gs_text, f"Net{net_idx}", tmpdir)
            if result is None:
                continue

            n_success += 1
            mse = compute_ko_kd_mse(result, actual)

            if mse < best_mse:
                best_mse = mse
                # Save SBML for potential later use
                if best_result and "sbml_content" in best_result:
                    pass
                best_result = dict(result)
                if os.path.exists(result["sbml_path"]):
                    best_result["sbml_content"] = open(result["sbml_path"]).read()

        if (trial + 1) % 50 == 0:
            log.info(f"    Trial {trial+1}/{n_trials}: {n_success} successes, best MSE={best_mse:.6f}")

    log.info(f"  Net {net_idx}: best MSE={best_mse:.6f} from {n_success}/{n_trials} successful trials")
    return best_result, best_mse


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------
def compare_data_type(gen: np.ndarray, act: np.ndarray, name: str) -> dict:
    """Compare generated vs actual expression for one data type."""
    if gen.size == 0 or act.size == 0:
        return {"name": name, "error": "empty data"}

    metrics = {"name": name, "gen_shape": gen.shape, "act_shape": act.shape}

    # Overall statistics
    metrics["gen_mean"] = float(gen.mean())
    metrics["act_mean"] = float(act.mean())
    metrics["gen_std"] = float(gen.std())
    metrics["act_std"] = float(act.std())
    metrics["gen_range"] = (float(gen.min()), float(gen.max()))
    metrics["act_range"] = (float(act.min()), float(act.max()))

    # MSE
    if gen.shape == act.shape:
        metrics["mse"] = float(np.mean((gen - act) ** 2))
        metrics["mae"] = float(np.mean(np.abs(gen - act)))
        # Per-gene MSE
        per_gene_mse = np.mean((gen - act) ** 2, axis=0)
        metrics["per_gene_mse"] = [float(x) for x in per_gene_mse]

    # KS test on flattened distributions
    ks_stat, ks_p = stats.ks_2samp(gen.flatten(), act.flatten())
    metrics["ks_stat"] = float(ks_stat)
    metrics["ks_pvalue"] = float(ks_p)

    # Correlation matrix similarity
    if gen.shape[0] > 2 and act.shape[0] > 2:
        try:
            gen_corr = np.corrcoef(gen.T)
            act_corr = np.corrcoef(act.T)
            frob = float(np.linalg.norm(gen_corr - act_corr, "fro"))
            metrics["corr_frobenius"] = frob
        except Exception:
            pass

    return metrics


def print_comparison_table(net_idx: int, calibrated: dict, actual: dict, mse: float):
    """Print detailed comparison for one network."""
    print(f"\n{'='*80}")
    print(f"Network {net_idx} — Calibrated MSE: {mse:.6f}")
    print(f"{'='*80}")

    for dtype, label in [("wt", "Wild-type"), ("ko", "Knockouts"),
                          ("kd", "Knockdowns"), ("mf", "Multifactorial"),
                          ("ts", "Timeseries")]:
        gen = calibrated.get(dtype)
        act = actual.get(dtype)
        if gen is None or act is None:
            print(f"\n  {label}: MISSING")
            continue

        is_target = dtype in ("ko", "kd")
        marker = " [CALIBRATION TARGET]" if is_target else " [HELD-OUT]"
        m = compare_data_type(gen, act, label)

        print(f"\n  {label}{marker}")
        print(f"    Shape: gen={m.get('gen_shape')} act={m.get('act_shape')}")
        print(f"    Mean:  gen={m['gen_mean']:.4f}  act={m['act_mean']:.4f}  Δ={abs(m['gen_mean']-m['act_mean']):.4f}")
        print(f"    Std:   gen={m['gen_std']:.4f}  act={m['act_std']:.4f}")
        print(f"    Range: gen=[{m['gen_range'][0]:.3f}, {m['gen_range'][1]:.3f}]  "
              f"act=[{m['act_range'][0]:.3f}, {m['act_range'][1]:.3f}]")
        print(f"    KS distance: {m['ks_stat']:.4f} (p={m['ks_pvalue']:.4e})")

        if "mse" in m:
            print(f"    MSE: {m['mse']:.6f}  MAE: {m['mae']:.4f}")

        if "corr_frobenius" in m:
            print(f"    Correlation matrix Frobenius: {m['corr_frobenius']:.4f}")

        if "per_gene_mse" in m and dtype in ("ko", "kd", "wt"):
            print(f"    Per-gene MSE: {' '.join(f'{x:.4f}' for x in m['per_gene_mse'])}")

    # Detailed per-gene comparison for WT
    if "wt" in calibrated and "wt" in actual:
        gen_wt = calibrated["wt"][0]
        act_wt = actual["wt"][0]
        print(f"\n  Per-gene Wild-type Comparison:")
        print(f"    {'Gene':<5} {'Actual':>10} {'Generated':>10} {'Δ':>10}")
        for i, g in enumerate(GENE_NAMES):
            print(f"    {g:<5} {act_wt[i]:>10.4f} {gen_wt[i]:>10.4f} {abs(gen_wt[i]-act_wt[i]):>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Validate GNW pipeline against DREAM4-10")
    parser.add_argument("--n_trials", type=int, default=500, help="GNW runs per network")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("DREAM4-10 GNW Pipeline Validation")
    log.info("=" * 60)

    # Verify data
    for net_idx in range(1, 6):
        d = DREAM4_COMPLETE / f"insilico_size10_{net_idx}"
        for suffix in ["wildtype", "knockouts", "knockdowns", "multifactorial", "timeseries"]:
            f = d / f"insilico_size10_{net_idx}_{suffix}.tsv"
            assert f.exists(), f"Missing: {f}"
    log.info("All 5 DREAM4-10 complete datasets verified")

    for net_idx in range(1, 6):
        gs = load_signed_goldstandard(net_idx)
        n_edges = len([l for l in gs.strip().split("\n") if l.strip()])
        log.info(f"  Net {net_idx}: {n_edges} signed edges")

    # Calibrate each network
    all_results = {}
    all_mses = {}
    summary_metrics = {}

    for net_idx in range(1, 6):
        log.info(f"\n{'='*60}")
        log.info(f"Processing Network {net_idx}")
        log.info(f"{'='*60}")

        calibrated, mse = calibrate_network(net_idx, n_trials=args.n_trials)
        actual = load_dream4_network(net_idx)

        if calibrated is None:
            log.warning(f"  Net {net_idx}: FAILED to calibrate")
            continue

        all_results[net_idx] = calibrated
        all_mses[net_idx] = mse

        # Print detailed comparison
        print_comparison_table(net_idx, calibrated, actual, mse)

        # Collect summary metrics
        net_metrics = {}
        for dtype in ["wt", "ko", "kd", "mf", "ts"]:
            gen = calibrated.get(dtype)
            act = actual.get(dtype)
            if gen is not None and act is not None:
                m = compare_data_type(gen, act, dtype)
                net_metrics[dtype] = m
        summary_metrics[net_idx] = net_metrics

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Per-Data-Type KS Distances (lower = more aligned)")
    print(f"{'='*80}")
    print(f"{'Net':<5} {'WT KS':>8} {'KO KS':>8} {'KD KS':>8} {'MF KS':>8} {'TS KS':>8} {'Cal MSE':>10}")
    for net_idx in range(1, 6):
        if net_idx not in summary_metrics:
            print(f"  {net_idx}    FAILED")
            continue
        m = summary_metrics[net_idx]
        vals = []
        for dtype in ["wt", "ko", "kd", "mf", "ts"]:
            if dtype in m:
                vals.append(f"{m[dtype]['ks_stat']:.4f}")
            else:
                vals.append("  N/A ")
        print(f"  {net_idx}  {'  '.join(vals)}  {all_mses.get(net_idx, float('nan')):.6f}")

    # MSE summary
    print(f"\n{'='*80}")
    print("SUMMARY: Per-Data-Type MSE (lower = more aligned)")
    print(f"{'='*80}")
    print(f"{'Net':<5} {'KO MSE':>10} {'KD MSE':>10} {'MF MSE':>10} {'WT MSE':>10}")
    for net_idx in range(1, 6):
        if net_idx not in summary_metrics:
            continue
        m = summary_metrics[net_idx]
        vals = []
        for dtype in ["ko", "kd", "mf", "wt"]:
            if dtype in m and "mse" in m[dtype]:
                vals.append(f"{m[dtype]['mse']:.6f}")
            else:
                vals.append("     N/A  ")
        print(f"  {net_idx}  {'  '.join(vals)}")

    # Average across networks
    if summary_metrics:
        print(f"\n{'='*80}")
        print("AVERAGE across networks")
        print(f"{'='*80}")
        for dtype, label in [("ko", "Knockouts [TARGET]"), ("kd", "Knockdowns [TARGET]"),
                              ("wt", "Wild-type [HELD-OUT]"), ("mf", "Multifactorial [HELD-OUT]"),
                              ("ts", "Timeseries [HELD-OUT]")]:
            ks_vals = []
            mse_vals = []
            for net_idx in summary_metrics:
                if dtype in summary_metrics[net_idx]:
                    m = summary_metrics[net_idx][dtype]
                    ks_vals.append(m["ks_stat"])
                    if "mse" in m:
                        mse_vals.append(m["mse"])
            if ks_vals:
                print(f"  {label:<30} KS={np.mean(ks_vals):.4f}±{np.std(ks_vals):.4f}"
                      f"  MSE={np.mean(mse_vals):.6f}±{np.std(mse_vals):.6f}" if mse_vals else
                      f"  {label:<30} KS={np.mean(ks_vals):.4f}±{np.std(ks_vals):.4f}")

    log.info("\nDone!")


if __name__ == "__main__":
    main()
