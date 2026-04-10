#!/usr/bin/env python3
"""Hybrid random-search + CMA-ES calibration of GNW kinetic parameters
to match DREAM4-10 expression matrices.

Strategy (fixes pure-CMA-ES failure):
  Each GNW --transform creates a unique SBML with both:
    (a) Boolean regulatory logic (numActivators, bindsAsComplex) — discrete, large effect
    (b) Continuous kinetic params (max, delta, k, n, a) — continuous, fine-tuning
  Pure CMA-ES on one template can only tune (b), but (a) matters enormously.

  Hybrid approach:
    Phase 1: Generate N random templates (diverse boolean logic), score each
    Phase 2: CMA-ES local refinement on top-K templates (small sigma from template values)
    Phase 3: Pick overall best

Usage:
    python scripts/calibrate_dream4_cmaes.py [--net_id 1] [--n_templates 200]
    python scripts/calibrate_dream4_cmaes.py  # all 5 networks
"""

import argparse
import glob
import logging
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import cma
import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GNW_JAR = str(PROJECT_ROOT / "data" / "gnw" / "gnw-3.1.2b.jar")
SETTINGS = str(PROJECT_ROOT / "data" / "gnw" / "settings_dream4_size10.txt")
DREAM4_DIR = Path("/tmp/dream4_complete")
GS_DIR = Path("/tmp/dream4_gs")

SBML_NS = {"s": "http://www.sbml.org/sbml/level2"}

# Parameter bounds (from observed GNW ranges)
PARAM_BOUNDS = {
    "max": (0.005, 0.15),
    "delta": (0.005, 0.15),
    "deltaProtein": (0.005, 0.15),
    "maxTranslation": (0.005, 0.15),
    "a_0": (0.001, 1.0),
    "k": (0.01, 1.0),
    "n": (1.0, 10.0),
    "a": (0.0, 1.0),
}

# GNW output suffixes to clean
GNW_OUTPUT_SUFFIXES = [
    "_knockouts.tsv", "_knockdowns.tsv", "_wildtype.tsv",
    "_multifactorial.tsv", "_dream4_timeseries.tsv",
    "_goldstandard.tsv", "_goldstandard_signed.tsv",
    "_nonoise_knockouts.tsv", "_nonoise_knockdowns.tsv",
    "_nonoise_wildtype.tsv", "_nonoise_multifactorial.tsv",
    "_nonoise_dream4_timeseries.tsv",
]


def load_tsv(path):
    with open(path) as f:
        header = [h.strip().strip('"') for h in f.readline().strip().split("\t")]
    data = np.loadtxt(path, skiprows=1, delimiter="\t")
    return header, data


def load_timeseries(path):
    with open(path) as f:
        header = [h.strip().strip('"') for h in f.readline().strip().split("\t")]
    rows = []
    with open(path) as f:
        f.readline()
        for line in f:
            if line.strip():
                rows.append([float(x) for x in line.strip().split("\t")])
    return header, np.array(rows)


def reorder(gen_genes, gen_data, act_genes, is_perturbation=False):
    """Reorder generated data to match actual gene order.
    For KO/KD: also reorder rows (row i = perturbation of gene i).
    """
    col_idx = [gen_genes.index(g) for g in act_genes]
    if is_perturbation:
        return gen_data[col_idx][:, col_idx]
    return gen_data[:, col_idx]


def run_gnw(args):
    return subprocess.run(
        ["java", "-jar", GNW_JAR] + args,
        capture_output=True, text=True, timeout=120, cwd="/tmp",
    )


def _clean_gnw_output(base_name):
    """Remove GNW output files for a given base name, without touching the SBML."""
    for suffix in GNW_OUTPUT_SUFFIXES:
        p = f"/tmp/{base_name}{suffix}"
        if os.path.exists(p):
            os.remove(p)


def generate_template(topo_path, template_name):
    """Run GNW --transform to create SBML with random kinetic params.

    GNW names output after input filename, so we copy the topology to
    /tmp/{template_name}.tsv before running --transform.
    """
    named_path = f"/tmp/{template_name}.tsv"
    shutil.copy2(topo_path, named_path)
    run_gnw([
        "--transform", "-c", SETTINGS,
        "--input-net", named_path,
        "--output-net-format=4", "--output-path", "/tmp",
    ])
    sbml_path = f"/tmp/{template_name}.xml"
    if not os.path.exists(sbml_path):
        raise RuntimeError(f"Transform failed: {sbml_path} not created")
    return sbml_path


def extract_params(sbml_path):
    """Extract continuous kinetic parameters from SBML as a flat vector.
    Returns: (param_vector, param_info) where param_info = [(reaction_id, param_id), ...]
    """
    tree = ET.parse(sbml_path)
    root = tree.getroot()
    params, info = [], []
    for reaction in root.findall(".//s:reaction", SBML_NS):
        rid = reaction.get("id")
        for param in reaction.findall(".//s:parameter", SBML_NS):
            pid = param.get("id")
            if pid.startswith("num") or pid.startswith("bindsAs"):
                continue
            params.append(float(param.get("value")))
            info.append((rid, pid))
    return np.array(params, dtype=np.float64), info


def get_param_bounds(info):
    """Get lower/upper bounds for each parameter."""
    lowers, uppers = [], []
    for _, pid in info:
        base = pid.rstrip("0123456789").rstrip("_") if "_" in pid else pid
        if base in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[base]
        else:
            lo, hi = 0.001, 1.0
        lowers.append(lo)
        uppers.append(hi)
    return np.array(lowers), np.array(uppers)


def write_modified_sbml(template_path, output_path, param_vector, info):
    """Write SBML with modified kinetic parameters."""
    tree = ET.parse(template_path)
    root = tree.getroot()
    lookup = {(rid, pid): val for (rid, pid), val in zip(info, param_vector)}
    for reaction in root.findall(".//s:reaction", SBML_NS):
        rid = reaction.get("id")
        for param in reaction.findall(".//s:parameter", SBML_NS):
            pid = param.get("id")
            key = (rid, pid)
            if key in lookup:
                param.set("value", f"{lookup[key]:.10f}")
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)


def simulate_sbml(sbml_path):
    """Run GNW --simulate on an SBML file. Returns the base name for output files."""
    base = os.path.splitext(os.path.basename(sbml_path))[0]
    _clean_gnw_output(base)
    run_gnw([
        "--simulate", "-c", SETTINGS,
        "--input-net", sbml_path,
        "--output-path", "/tmp",
    ])
    return base


def score_simulation(base_name, act_ko, act_kd, act_genes):
    """Score a simulation's KO+KD output against actual DREAM4 data."""
    ko_path = f"/tmp/{base_name}_knockouts.tsv"
    kd_path = f"/tmp/{base_name}_knockdowns.tsv"
    if not os.path.exists(ko_path) or not os.path.exists(kd_path):
        return float("inf")
    try:
        gen_genes_ko, gen_ko = load_tsv(ko_path)
        gen_genes_kd, gen_kd = load_tsv(kd_path)
    except Exception:
        return float("inf")
    if gen_ko.ndim == 1:
        gen_ko = gen_ko.reshape(1, -1)
    if gen_kd.ndim == 1:
        gen_kd = gen_kd.reshape(1, -1)
    try:
        gen_ko = reorder(gen_genes_ko, gen_ko, act_genes, is_perturbation=True)
        gen_kd = reorder(gen_genes_kd, gen_kd, act_genes, is_perturbation=True)
    except (ValueError, IndexError):
        return float("inf")

    # MSE excluding diagonal zeros in KO (knocked-out gene always 0)
    ko_mask = np.ones_like(act_ko, dtype=bool)
    for i in range(min(act_ko.shape)):
        ko_mask[i, i] = False
    mse_ko = np.mean((gen_ko[ko_mask] - act_ko[ko_mask]) ** 2)
    mse_kd = np.mean((gen_kd - act_kd) ** 2)
    return (mse_ko + mse_kd) / 2


def simulate_and_score_sbml(sbml_path, act_ko, act_kd, act_genes):
    """Simulate + score in one call. Uses SBML filename as output prefix."""
    base = simulate_sbml(sbml_path)
    mse = score_simulation(base, act_ko, act_kd, act_genes)
    _clean_gnw_output(base)
    return mse


def compute_detailed_metrics(sbml_path, actual, act_genes):
    """Compute comprehensive alignment metrics for the best solution."""
    base = simulate_sbml(sbml_path)

    results = {}
    for dtype, suffix, is_ts, is_perturb in [
        ("wt", "_wildtype.tsv", False, False),
        ("ko", "_knockouts.tsv", False, True),
        ("kd", "_knockdowns.tsv", False, True),
        ("mf", "_multifactorial.tsv", False, False),
        ("ts", "_dream4_timeseries.tsv", True, False),
    ]:
        path = f"/tmp/{base}{suffix}"
        if not os.path.exists(path):
            continue

        if is_ts:
            gen_g, gen_d = load_timeseries(path)
            if "Time" in gen_g:
                ti = gen_g.index("Time")
                gen_g = gen_g[:ti] + gen_g[ti+1:]
                gen_d = np.delete(gen_d, ti, axis=1)
        else:
            gen_g, gen_d = load_tsv(path)
        if gen_d.ndim == 1:
            gen_d = gen_d.reshape(1, -1)

        act_d = actual[dtype]
        try:
            gen_d = reorder(gen_g, gen_d, act_genes, is_perturbation=is_perturb)
        except (ValueError, IndexError):
            continue

        r_elem, _ = stats.pearsonr(act_d.flatten(), gen_d.flatten())
        mse = float(np.mean((act_d - gen_d) ** 2))
        r_prof, _ = stats.pearsonr(act_d.mean(0), gen_d.mean(0))

        metrics = {"elem_r": r_elem, "mse": mse, "prof_r": r_prof}

        if act_d.shape[0] > 2:
            grs = []
            for g in range(act_d.shape[1]):
                if np.std(act_d[:, g]) > 1e-10 and np.std(gen_d[:, g]) > 1e-10:
                    r, _ = stats.pearsonr(act_d[:, g], gen_d[:, g])
                    grs.append(r)
            if grs:
                metrics["per_gene_r"] = float(np.mean(grs))

        if act_d.shape[0] > 2:
            ac = np.corrcoef(act_d.T)
            gc = np.corrcoef(gen_d.T)
            mask_idx = np.triu_indices(act_d.shape[1], k=1)
            r_gg, _ = stats.pearsonr(ac[mask_idx], gc[mask_idx])
            metrics["gene_gene_r"] = float(r_gg)

        results[dtype] = metrics

    _clean_gnw_output(base)
    return results


def load_dream4_actual(net_id):
    """Load actual DREAM4-10 expression data."""
    d = DREAM4_DIR / f"insilico_size10_{net_id}"
    actual = {}
    genes = None

    for dtype, fname, is_ts in [
        ("wt", f"insilico_size10_{net_id}_wildtype.tsv", False),
        ("ko", f"insilico_size10_{net_id}_knockouts.tsv", False),
        ("kd", f"insilico_size10_{net_id}_knockdowns.tsv", False),
        ("mf", f"insilico_size10_{net_id}_multifactorial.tsv", False),
        ("ts", f"insilico_size10_{net_id}_timeseries.tsv", True),
    ]:
        path = str(d / fname)
        if is_ts:
            g, data = load_timeseries(path)
            # Strip "Time" column — keep only gene expression columns
            if "Time" in g:
                ti = g.index("Time")
                g = g[:ti] + g[ti+1:]
                data = np.delete(data, ti, axis=1)
        else:
            g, data = load_tsv(path)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        actual[dtype] = data
        if genes is None:
            genes = g

    return actual, genes


def convert_gs_to_gnw_input(gs_path, output_path):
    """Convert unsigned gold standard (tf target 1/0) to GNW edge list."""
    edges = []
    with open(gs_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 3 and parts[2] == "1":
                edges.append(f"{parts[0]}\t{parts[1]}")
    with open(output_path, "w") as f:
        f.write("\n".join(edges) + "\n")
    return len(edges)


def cmaes_refine(template_sbml, act_ko, act_kd, act_genes,
                 max_evals=200, sigma0=0.15, seed=42):
    """Run CMA-ES local refinement starting from a template's kinetic parameters.

    Returns (best_mse, best_sbml_path).
    """
    x0, param_info = extract_params(template_sbml)
    lowers, uppers = get_param_bounds(param_info)
    n_params = len(x0)
    ranges = uppers - lowers

    # Normalize template params to [0,1]
    x0_norm = np.clip((x0 - lowers) / ranges, 0.01, 0.99)

    # Use a dedicated SBML path for CMA-ES evaluations
    work_sbml = template_sbml.replace(".xml", "_cma_work.xml")
    best_sbml = template_sbml.replace(".xml", "_cma_best.xml")

    # Evaluate template as starting point
    baseline_mse = simulate_and_score_sbml(template_sbml, act_ko, act_kd, act_genes)
    best_mse = baseline_mse
    best_x_norm = x0_norm.copy()

    if np.isinf(best_mse):
        # Template itself fails to simulate — skip
        return float("inf"), None

    opts = {
        "seed": seed,
        "maxfevals": max_evals,
        "bounds": [0.001, 0.999],
        "tolfun": 1e-9,
        "tolx": 1e-7,
        "verb_disp": 0,
        "verb_log": 0,
        "verbose": -9,
        "popsize": max(10, 4 + int(3 * np.log(n_params))),
    }

    es = cma.CMAEvolutionStrategy(x0_norm, sigma0, opts)
    n_eval = 0
    n_inf = 0

    while not es.stop() and n_eval < max_evals:
        solutions = es.ask()
        fitnesses = []
        for x_norm in solutions:
            x_clipped = np.clip(x_norm, 0.001, 0.999)
            x_real = x_clipped * ranges + lowers
            write_modified_sbml(template_sbml, work_sbml, x_real, param_info)
            mse = simulate_and_score_sbml(work_sbml, act_ko, act_kd, act_genes)
            # Use large finite penalty for failed simulations to give CMA-ES gradient signal
            if np.isinf(mse):
                mse = 10.0
                n_inf += 1
            fitnesses.append(mse)
            n_eval += 1
            if mse < best_mse:
                best_mse = mse
                best_x_norm = x_clipped.copy()
        es.tell(solutions, fitnesses)

    # Write best result
    best_x_real = best_x_norm * ranges + lowers
    write_modified_sbml(template_sbml, best_sbml, best_x_real, param_info)

    # Clean work file
    if os.path.exists(work_sbml):
        os.remove(work_sbml)

    return best_mse, best_sbml


def calibrate_network(net_id, n_templates=200, n_refine_top=10,
                      cma_evals_per_template=200, seed=42):
    """Hybrid random-search + CMA-ES calibration for one DREAM4-10 network."""
    log.info(f"{'='*70}")
    log.info(f"Calibrating Network {net_id}")
    log.info(f"  Phase 1: {n_templates} random templates")
    log.info(f"  Phase 2: CMA-ES refine top-{n_refine_top} ({cma_evals_per_template} evals each)")
    log.info(f"{'='*70}")

    np.random.seed(seed)

    # Load actual DREAM4 data
    actual, act_genes = load_dream4_actual(net_id)
    act_ko = actual["ko"]
    act_kd = actual["kd"]

    # Create GNW input topology
    gs_path = str(GS_DIR / f"gs_{net_id}.tsv")
    topo_path = f"/tmp/cma_net{net_id}_topo.tsv"
    n_edges = convert_gs_to_gnw_input(gs_path, topo_path)
    log.info(f"  Topology: {n_edges} edges")

    # ── Phase 1: Random template search ──
    log.info(f"\n  Phase 1: Generating {n_templates} random templates...")
    template_scores = []  # [(mse, sbml_path)]

    for i in range(n_templates):
        tname = f"cma_n{net_id}_t{i:04d}"
        try:
            sbml = generate_template(topo_path, tname)
        except RuntimeError:
            continue

        mse = simulate_and_score_sbml(sbml, act_ko, act_kd, act_genes)
        if not np.isinf(mse):
            template_scores.append((mse, sbml))

        if (i + 1) % 50 == 0:
            valid = len(template_scores)
            best_so_far = min(s[0] for s in template_scores) if template_scores else float("inf")
            log.info(f"    {i+1}/{n_templates}: {valid} valid, best MSE={best_so_far:.6f}")

    if not template_scores:
        log.error("  No valid templates found!")
        return {"net_id": net_id, "best_mse": float("inf"), "metrics": {}}

    # Sort and keep top-K
    template_scores.sort(key=lambda x: x[0])
    n_valid = len(template_scores)
    top_k = min(n_refine_top, n_valid)
    top_templates = template_scores[:top_k]

    log.info(f"\n  Phase 1 done: {n_valid}/{n_templates} valid templates")
    log.info(f"  Top-{top_k} MSE: {[f'{s:.5f}' for s, _ in top_templates]}")
    phase1_best_mse = top_templates[0][0]
    phase1_best_sbml = top_templates[0][1]

    # ── Phase 2: CMA-ES local refinement on top-K templates ──
    log.info(f"\n  Phase 2: CMA-ES refinement on top-{top_k} templates...")
    overall_best_mse = phase1_best_mse
    overall_best_sbml = phase1_best_sbml

    for rank, (tmpl_mse, tmpl_sbml) in enumerate(top_templates):
        log.info(f"    Refining rank-{rank+1} template (MSE={tmpl_mse:.6f})...")
        refined_mse, refined_sbml = cmaes_refine(
            tmpl_sbml, act_ko, act_kd, act_genes,
            max_evals=cma_evals_per_template,
            sigma0=0.15,
            seed=seed + rank,
        )
        improvement = (1 - refined_mse / tmpl_mse) * 100 if tmpl_mse > 0 else 0
        log.info(f"      {tmpl_mse:.6f} → {refined_mse:.6f} ({improvement:+.1f}%)")

        if refined_mse < overall_best_mse and refined_sbml is not None:
            overall_best_mse = refined_mse
            overall_best_sbml = refined_sbml

    # Copy best SBML to stable path
    best_final = f"/tmp/cma_net{net_id}_best.xml"
    shutil.copy2(overall_best_sbml, best_final)

    log.info(f"\n  Overall best: MSE={overall_best_mse:.6f} "
             f"(Phase 1 best was {phase1_best_mse:.6f}, "
             f"improvement {(1 - overall_best_mse/phase1_best_mse)*100:+.1f}%)")

    # ── Phase 3: Detailed metrics ──
    log.info("  Computing detailed alignment metrics...")
    metrics = compute_detailed_metrics(best_final, actual, act_genes)

    log.info(f"\n  {'─'*60}")
    log.info(f"  Network {net_id} — Best MSE: {overall_best_mse:.6f}")
    log.info(f"  {'─'*60}")
    for dtype, m in metrics.items():
        label = {"wt": "Wild-type", "ko": "Knockouts", "kd": "Knockdowns",
                 "mf": "Multifactorial", "ts": "Timeseries"}.get(dtype, dtype)
        role = "TARGET" if dtype in ("ko", "kd") else "HELD-OUT"
        parts = [f"elem_r={m['elem_r']:.3f}", f"MSE={m['mse']:.4f}",
                 f"prof_r={m['prof_r']:.3f}"]
        if "per_gene_r" in m:
            parts.append(f"per_gene_r={m['per_gene_r']:.3f}")
        if "gene_gene_r" in m:
            parts.append(f"gene_gene_r={m['gene_gene_r']:.3f}")
        log.info(f"  {label:15s} [{role:8s}]: {', '.join(parts)}")

    # Clean up intermediate templates (keep only best)
    for f in glob.glob(f"/tmp/cma_n{net_id}_t*"):
        if f != best_final:
            os.remove(f)

    return {
        "net_id": net_id,
        "best_mse": overall_best_mse,
        "phase1_best_mse": phase1_best_mse,
        "n_templates": n_templates,
        "n_valid": n_valid,
        "best_sbml": best_final,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid random-search + CMA-ES kinetic parameter calibration"
    )
    parser.add_argument("--net_id", type=int, default=0,
                        help="Network ID (1-5), or 0 for all")
    parser.add_argument("--n_templates", type=int, default=200,
                        help="Number of random templates in Phase 1")
    parser.add_argument("--n_refine_top", type=int, default=10,
                        help="Number of top templates to refine with CMA-ES")
    parser.add_argument("--cma_evals", type=int, default=200,
                        help="CMA-ES evaluations per template in Phase 2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nets = [args.net_id] if args.net_id > 0 else list(range(1, 6))

    all_results = {}
    for net_id in nets:
        result = calibrate_network(
            net_id,
            n_templates=args.n_templates,
            n_refine_top=args.n_refine_top,
            cma_evals_per_template=args.cma_evals,
            seed=args.seed,
        )
        all_results[net_id] = result

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY: Hybrid Random + CMA-ES Calibration Results")
    print("=" * 80)

    print(f"\n{'Net':>4} {'Templates':>10} {'Valid':>6} {'Ph1 MSE':>10} "
          f"{'Best MSE':>10} {'CMA Improv':>11}")
    for net_id, r in sorted(all_results.items()):
        improv = (1 - r["best_mse"] / r["phase1_best_mse"]) * 100
        print(f"{net_id:>4} {r['n_templates']:>10} {r.get('n_valid', '?'):>6} "
              f"{r['phase1_best_mse']:>10.6f} {r['best_mse']:>10.6f} {improv:>+10.1f}%")

    print(f"\n{'Net':>4} {'KO elem_r':>10} {'KD elem_r':>10} {'WT elem_r':>10} "
          f"{'MF elem_r':>10} {'TS elem_r':>10}")
    for net_id, r in sorted(all_results.items()):
        m = r["metrics"]
        vals = [m.get(d, {}).get("elem_r", float("nan"))
                for d in ["ko", "kd", "wt", "mf", "ts"]]
        print(f"{net_id:>4}", "".join(f"{v:>10.3f}" for v in vals))

    print(f"\n{'Net':>4} {'KO gene_r':>10} {'KD gene_r':>10} {'KO gg_r':>10}")
    for net_id, r in sorted(all_results.items()):
        m = r["metrics"]
        ko_gr = m.get("ko", {}).get("per_gene_r", float("nan"))
        kd_gr = m.get("kd", {}).get("per_gene_r", float("nan"))
        ko_gg = m.get("ko", {}).get("gene_gene_r", float("nan"))
        print(f"{net_id:>4} {ko_gr:>10.3f} {kd_gr:>10.3f} {ko_gg:>10.3f}")

    # Compare with previous random search baseline (200 trials, no CMA-ES)
    print("\n" + "=" * 80)
    print("Comparison: Previous Random Search (200 trials) vs Hybrid")
    print("=" * 80)
    random_baselines = {
        1: {"mse": 0.012, "ko_r": 0.932, "kd_r": 0.889},
        2: {"mse": 0.023, "ko_r": 0.872, "kd_r": 0.835},
        3: {"mse": 0.026, "ko_r": 0.837, "kd_r": 0.637},
        4: {"mse": 0.018, "ko_r": 0.879, "kd_r": 0.892},
        5: {"mse": 0.018, "ko_r": 0.934, "kd_r": 0.884},
    }
    print(f"{'Net':>4} {'Prev MSE':>10} {'Hybrid MSE':>11} {'Prev KO_r':>10} {'Hybrid KO_r':>12}")
    for net_id, r in sorted(all_results.items()):
        rb = random_baselines.get(net_id, {})
        hybrid_ko = r["metrics"].get("ko", {}).get("elem_r", float("nan"))
        print(f"{net_id:>4} {rb.get('mse', float('nan')):>10.4f} "
              f"{r['best_mse']:>11.6f} "
              f"{rb.get('ko_r', float('nan')):>10.3f} {hybrid_ko:>12.3f}")


if __name__ == "__main__":
    main()
