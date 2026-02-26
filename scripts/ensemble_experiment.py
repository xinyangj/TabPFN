"""Ensemble experiment: Rank-average TabPFN IG with baseline methods on DREAM4-10.

Tests whether combining TabPFN IG edge scores with baseline methods via
rank averaging improves GRN inference AUPR/AUROC.

Combinations tested:
  - IG + GENIE3
  - IG + GRNBoost2
  - IG + Correlation
  - IG + Mutual Info
  - IG + All four baselines (full ensemble)
"""

from __future__ import annotations

import gc
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNBaselineRunner,
    GRNPreprocessor,
    evaluate_grn,
)


def cleanup_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def rank_average_edge_scores(
    score_dicts: list[dict[tuple[str, str], float]],
) -> dict[tuple[str, str], float]:
    """Rank-average multiple edge score dictionaries.

    Each method's scores are converted to ranks (1 = highest score),
    then ranks are averaged across methods. The final score is
    negative average rank (so higher = better, matching AUPR convention).
    """
    all_edges = set()
    for d in score_dicts:
        all_edges.update(d.keys())
    all_edges = sorted(all_edges)

    n_methods = len(score_dicts)
    rank_sum = {e: 0.0 for e in all_edges}

    for d in score_dicts:
        # Sort edges by score descending, assign ranks
        edges_sorted = sorted(all_edges, key=lambda e: d.get(e, 0.0), reverse=True)
        for rank_idx, edge in enumerate(edges_sorted):
            rank_sum[edge] += rank_idx + 1  # rank 1 = best

    # Average rank → convert to score (lower rank = higher score)
    max_rank = len(all_edges)
    return {e: (max_rank - rank_sum[e] / n_methods) for e in all_edges}


def zscore_average_edge_scores(
    score_dicts: list[dict[tuple[str, str], float]],
) -> dict[tuple[str, str], float]:
    """Z-score normalize each method's scores, then average.

    Preserves magnitude information (unlike rank averaging).
    Each method's scores are standardized to zero-mean, unit-variance
    before averaging.
    """
    all_edges = set()
    for d in score_dicts:
        all_edges.update(d.keys())
    all_edges = sorted(all_edges)

    combined = {e: 0.0 for e in all_edges}

    for d in score_dicts:
        scores = np.array([d.get(e, 0.0) for e in all_edges])
        std = scores.std()
        if std > 0:
            scores = (scores - scores.mean()) / std
        else:
            scores = scores - scores.mean()
        for i, e in enumerate(all_edges):
            combined[e] += scores[i]

    n_methods = len(score_dicts)
    return {e: combined[e] / n_methods for e in all_edges}


def run_all_methods_on_network(
    expression, gene_names, tf_names, gold_standard, dataset_name,
    ig_n_folds=5, ig_baseline="mean", rise_n_folds=5,
):
    """Run all methods on one network, return edge scores dict per method."""
    runner = GRNBaselineRunner(normalization="zscore")
    edge_scores = {}

    # TabPFN IG + RISE (run together to share model loading)
    print(f"  Running TabPFN IG (folds={ig_n_folds}) + RISE (folds={rise_n_folds})...")
    t0 = time.time()
    tabpfn_results = runner.run_tabpfn_multiple_strategies(
        expression=expression,
        gene_names=gene_names,
        tf_names=tf_names,
        gold_standard=gold_standard,
        dataset_name=dataset_name,
        n_estimators=1,
        edge_score_strategies=["integrated_gradients", "rise"],
        ig_n_folds=ig_n_folds,
        ig_baseline=ig_baseline,
        rise_n_folds=rise_n_folds,
    )
    for strategy, result in tabpfn_results.items():
        label = "TabPFN_IG" if strategy == "integrated_gradients" else "TabPFN_RISE"
        edge_scores[label] = result["edge_scores"]
        print(f"    {label}: AUPR={result['metrics']['aupr']:.4f}")
    print(f"    TabPFN total: {time.time()-t0:.1f}s")
    cleanup_gpu_memory()

    # Baselines
    for method in ["correlation", "mutual_info", "genie3", "grnboost2"]:
        t0 = time.time()
        print(f"  Running {method}...")
        result = runner.run_method(
            method=method,
            expression=expression,
            gene_names=gene_names,
            tf_names=tf_names,
            gold_standard=gold_standard,
            dataset_name=dataset_name,
        )
        label = {
            "correlation": "Correlation",
            "mutual_info": "MutualInfo",
            "genie3": "GENIE3",
            "grnboost2": "GRNBoost2",
        }[method]
        edge_scores[label] = result["edge_scores"]
        print(f"    {label}: AUPR={result['metrics']['aupr']:.4f} ({time.time()-t0:.1f}s)")
        cleanup_gpu_memory()

    return edge_scores


def evaluate_ensembles(edge_scores, gold_standard):
    """Evaluate individual methods and ensembles using rank avg and z-score avg."""
    results = {}

    # Individual methods
    for name, scores in edge_scores.items():
        metrics = evaluate_grn(scores, gold_standard)
        results[name] = {"aupr": metrics["aupr"], "auroc": metrics["auroc"]}

    baselines = ["GENIE3", "GRNBoost2", "Correlation", "MutualInfo"]
    tabpfn_methods = ["TabPFN_IG", "TabPFN_RISE"]
    fusion_fns = {
        "rank": rank_average_edge_scores,
        "zscore": zscore_average_edge_scores,
    }

    for fusion_name, fusion_fn in fusion_fns.items():
        suffix = f"[{fusion_name}]"

        # Pairwise: IG + each baseline
        for bl in baselines:
            if "TabPFN_IG" in edge_scores and bl in edge_scores:
                name = f"IG+{bl} {suffix}"
                ens = fusion_fn([edge_scores["TabPFN_IG"], edge_scores[bl]])
                m = evaluate_grn(ens, gold_standard)
                results[name] = {"aupr": m["aupr"], "auroc": m["auroc"]}

        # IG + RISE (pure TabPFN)
        if "TabPFN_IG" in edge_scores and "TabPFN_RISE" in edge_scores:
            name = f"IG+RISE {suffix}"
            ens = fusion_fn([edge_scores["TabPFN_IG"], edge_scores["TabPFN_RISE"]])
            m = evaluate_grn(ens, gold_standard)
            results[name] = {"aupr": m["aupr"], "auroc": m["auroc"]}

        # IG + RISE + GENIE3 (3-way)
        if all(k in edge_scores for k in ["TabPFN_IG", "TabPFN_RISE", "GENIE3"]):
            name = f"IG+RISE+GENIE3 {suffix}"
            ens = fusion_fn([
                edge_scores["TabPFN_IG"],
                edge_scores["TabPFN_RISE"],
                edge_scores["GENIE3"],
            ])
            m = evaluate_grn(ens, gold_standard)
            results[name] = {"aupr": m["aupr"], "auroc": m["auroc"]}

        # IG + all baselines
        ig_plus_all = [edge_scores["TabPFN_IG"]]
        for bl in baselines:
            if bl in edge_scores:
                ig_plus_all.append(edge_scores[bl])
        if len(ig_plus_all) > 1:
            name = f"IG+AllBL {suffix}"
            ens = fusion_fn(ig_plus_all)
            m = evaluate_grn(ens, gold_standard)
            results[name] = {"aupr": m["aupr"], "auroc": m["auroc"]}

        # IG + RISE + all baselines (full 6-method)
        full = []
        for k in tabpfn_methods + baselines:
            if k in edge_scores:
                full.append(edge_scores[k])
        if len(full) > 1:
            name = f"All6 {suffix}"
            ens = fusion_fn(full)
            m = evaluate_grn(ens, gold_standard)
            results[name] = {"aupr": m["aupr"], "auroc": m["auroc"]}

        # Baselines-only ensemble
        bl_dicts = [edge_scores[bl] for bl in baselines if bl in edge_scores]
        if len(bl_dicts) > 1:
            name = f"BaselinesOnly {suffix}"
            ens = fusion_fn(bl_dicts)
            m = evaluate_grn(ens, gold_standard)
            results[name] = {"aupr": m["aupr"], "auroc": m["auroc"]}

    return results


def load_dream5_data(dream5_path, max_targets=30):
    """Load and preprocess DREAM5 E.coli data (same logic as grn_performance_analysis.py)."""
    import pandas as pd

    loader = DREAMChallengeLoader(data_path=str(dream5_path))
    expression, gene_names, tf_names, gold_standard = loader.load_dream5_ecoli()

    # Convert gold_standard DataFrame to set of positive edges only
    if isinstance(gold_standard, pd.DataFrame):
        if "weight" in gold_standard.columns:
            gold_standard = gold_standard[gold_standard["weight"] == 1]
        gs_set = set()
        for _, row in gold_standard.iterrows():
            if "tf" in row and "target" in row:
                gs_set.add((row["tf"], row["target"]))
            elif "TF" in row and "Target" in row:
                gs_set.add((row["TF"], row["Target"]))
            elif "source" in row and "target" in row:
                gs_set.add((row["source"], row["target"]))
            elif len(row) >= 2:
                gs_set.add((row.iloc[0], row.iloc[1]))
        gold_standard = gs_set

    # Filter to only genes in gold standard
    gold_genes = set()
    for tf, tgt in gold_standard:
        gold_genes.add(tf)
        gold_genes.add(tgt)

    gold_gene_list = list(gold_genes & set(gene_names))
    gene_idx_map = {g: i for i, g in enumerate(gene_names)}
    gold_indices = [gene_idx_map[g] for g in gold_gene_list]

    expression_filtered = expression[:, gold_indices]
    gene_names_filtered = gold_gene_list
    tf_names_filtered = [tf for tf in tf_names if tf in gold_gene_list]

    preprocessor = GRNPreprocessor(normalization="zscore")
    X, y, tf_indices, target_indices = preprocessor.fit_transform(
        expression_filtered, gene_names_filtered, tf_names_filtered
    )
    all_target_genes = preprocessor.get_target_names()
    target_genes = all_target_genes[:max_targets]

    filtered_gold_standard = {
        (tf, tgt) for tf, tgt in gold_standard
        if tgt in target_genes and tf in tf_names_filtered
    }

    # Create expression matrix with only TFs + selected targets for baseline runner
    tf_indices_filtered = [gene_names_filtered.index(tf) for tf in tf_names_filtered]
    target_indices_filtered = [gene_names_filtered.index(tgt) for tgt in target_genes]
    expression_for_baseline = expression_filtered[:, tf_indices_filtered + target_indices_filtered]
    gene_names_for_baseline = tf_names_filtered + target_genes

    return (
        expression_for_baseline,
        gene_names_for_baseline,
        tf_names_filtered,
        filtered_gold_standard,
        target_genes,
    )


def print_results_table(results, title):
    """Print a sorted results table."""
    print(f"\n{'Method':<30} {'AUPR':>8} {'AUROC':>8}")
    print(f"{'-'*46}")
    for method in sorted(results, key=lambda m: results[m]["aupr"], reverse=True):
        print(f"{method:<30} {results[method]['aupr']:>8.4f} {results[method]['auroc']:>8.4f}")


def print_aggregate_table(all_network_results, title):
    """Print aggregate results with std across networks."""
    print(f"\n\n{'='*70}")
    print(title)
    print(f"{'='*70}")

    all_methods = set()
    for net_results in all_network_results.values():
        all_methods.update(net_results.keys())

    avg_results = {}
    for method in sorted(all_methods):
        auprs = [
            all_network_results[net][method]["aupr"]
            for net in all_network_results
            if method in all_network_results[net]
        ]
        aurocs = [
            all_network_results[net][method]["auroc"]
            for net in all_network_results
            if method in all_network_results[net]
        ]
        if auprs:
            avg_results[method] = {
                "aupr": np.mean(auprs),
                "auroc": np.mean(aurocs),
                "aupr_std": np.std(auprs),
                "auroc_std": np.std(aurocs),
            }

    print(f"\n{'Method':<30} {'AUPR':>12} {'AUROC':>12}")
    print(f"{'-'*54}")
    for method in sorted(avg_results, key=lambda m: avg_results[m]["aupr"], reverse=True):
        r = avg_results[method]
        print(f"{method:<30} {r['aupr']:.4f}±{r['aupr_std']:.4f} {r['auroc']:.4f}±{r['auroc_std']:.4f}")


def run_dream4_10(dream4_path):
    """Run ensemble experiment on DREAM4-10 (5 networks)."""
    loader = DREAMChallengeLoader(data_path=str(dream4_path))
    all_network_results = {}

    for network_id in range(1, 6):
        net_name = f"DREAM4_10_{network_id}"
        print(f"\n{'='*70}")
        print(f"Network: {net_name}")
        print(f"{'='*70}")

        expression, gene_names, tf_names, gold_standard = loader.load_dream4(
            network_size=10, network_id=network_id
        )

        edge_scores = run_all_methods_on_network(
            expression, gene_names, tf_names, gold_standard, net_name,
            ig_n_folds=5, ig_baseline="mean", rise_n_folds=5,
        )
        results = evaluate_ensembles(edge_scores, gold_standard)
        all_network_results[net_name] = results
        print_results_table(results, net_name)

    print_aggregate_table(all_network_results, "AGGREGATE: DREAM4-10 (Mean across 5 networks)")
    return all_network_results


def run_dream5(dream5_path, max_targets=30):
    """Run ensemble experiment on DREAM5 E.coli."""
    print(f"\n{'='*70}")
    print(f"DREAM5 E.coli (max_targets={max_targets})")
    print(f"{'='*70}")

    cleanup_gpu_memory()

    expression, gene_names, tf_names, gold_standard, target_genes = load_dream5_data(
        dream5_path, max_targets=max_targets
    )

    edge_scores = run_all_methods_on_network(
        expression, gene_names, tf_names, gold_standard,
        "DREAM5_Ecoli",
        ig_n_folds=1,  # 1-fold for speed (DREAM5 is large)
        ig_baseline="mean",
        rise_n_folds=1,
    )
    results = evaluate_ensembles(edge_scores, gold_standard)
    print_results_table(results, "DREAM5 E.coli")
    return {"DREAM5_Ecoli": results}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ensemble experiment")
    parser.add_argument("--datasets", nargs="+", default=["dream4-10"],
                        choices=["dream4-10", "dream5", "all"])
    parser.add_argument("--max-targets", type=int, default=30,
                        help="Max targets for DREAM5")
    parser.add_argument("--ig-n-folds", type=int, default=None,
                        help="Override IG folds (default: 5 for D4-10, 1 for D5)")
    args = parser.parse_args()

    datasets = args.datasets
    if "all" in datasets:
        datasets = ["dream4-10", "dream5"]

    if "dream4-10" in datasets:
        run_dream4_10(Path("data/dream4"))

    if "dream5" in datasets:
        run_dream5(Path("data/dream5"), max_targets=args.max_targets)


if __name__ == "__main__":
    main()
