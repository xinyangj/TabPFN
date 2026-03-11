#!/usr/bin/env python
"""Comprehensive experiment: Train and evaluate the interpretation model.

Loads pre-generated datasets from disk (see generate_interpretation_data.py),
trains interpretation models (MLP and Transformer), evaluates across 4 label
modes, runs a feature ablation study, and produces a summary report.

Usage:
    # First generate data:
    python scripts/generate_interpretation_data.py --n_datasets 10000
    # Then run experiment:
    python scripts/run_interpretation_experiment.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/interpretation_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path("data/interpretation_cache")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# Phase 1: Load Data from Disk
# ============================================================

def load_data(cache_dir: Path = CACHE_DIR) -> list[dict]:
    """Load pre-generated datasets from disk cache."""
    import sys
    # Ensure project root is on path for script imports
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from scripts.generate_interpretation_data import load_all_datasets
    return load_all_datasets(cache_dir)


# ============================================================
# Phase 2: Train Interpretation Models
# ============================================================

def train_model(
    train_data: list[dict],
    val_data: list[dict],
    *,
    variant: str = "mlp",
    label_mode: str = "binary_direct",
    max_features: int = 30,
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
) -> tuple:
    """Train an interpretation model and return model + history."""
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.training.dataset import InterpretationDataset

    # Determine input dimension
    input_dim = train_data[0]["feature_vectors"].shape[1]

    # Create datasets
    train_ds = InterpretationDataset(
        train_data, label_mode=label_mode, max_features=max_features
    )
    val_ds = InterpretationDataset(
        val_data, label_mode=label_mode, max_features=max_features, augment=False
    )

    # Determine output mode
    output_mode = "binary" if label_mode in ("binary_direct", "binary_ancestry") else "continuous"

    # Create model
    if variant == "mlp":
        model = InterpretationModel(
            "mlp", input_dim=input_dim, output_mode=output_mode,
            hidden_dims=[512, 256, 128],
        )
    else:
        model = InterpretationModel(
            "transformer", input_dim=input_dim, output_mode=output_mode,
            d_model=256, n_heads=4, n_layers=2,
        )

    model = model.to(DEVICE)
    loss_fn = InterpretationLoss(mode=output_mode).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(n_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            preds = model(features, mask=mask)
            loss = loss_fn(preds, labels, mask=mask.bool())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # Validate
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                preds = model(features, mask=mask)
                loss = loss_fn(preds, labels, mask=mask.bool())
                val_loss += loss.item()
                n_val += 1

        val_loss = val_loss / max(n_val, 1)
        history["val_loss"].append(val_loss)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0:
            logger.info(f"  Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    return model, history


# ============================================================
# Phase 3: Evaluation
# ============================================================

def evaluate_model(
    model,
    test_data: list[dict],
    label_mode: str,
    max_features: int = 30,
) -> dict:
    """Evaluate model on test data."""
    from tabpfn.interpretation.evaluation.metrics import (
        evaluate_binary,
        evaluate_continuous,
        evaluate_ranking,
    )
    from tabpfn.interpretation.training.dataset import InterpretationDataset

    output_mode = "binary" if label_mode in ("binary_direct", "binary_ancestry") else "continuous"

    test_ds = InterpretationDataset(
        test_data, label_mode=label_mode, max_features=max_features, augment=False
    )

    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for i in range(len(test_ds)):
            batch = test_ds[i]
            features = batch["features"].unsqueeze(0).to(DEVICE)
            mask = batch["mask"].unsqueeze(0).to(DEVICE)
            labels = batch["labels"].numpy()

            preds = model(features, mask=mask)
            if output_mode == "binary":
                preds = torch.sigmoid(preds)
            preds = preds.cpu().numpy().squeeze(0)

            n_feat = int(batch["n_features"])
            all_preds.append(preds[:n_feat])
            all_targets.append(labels[:n_feat])

    # Aggregate metrics
    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)

    metrics = {}
    if output_mode == "binary":
        metrics.update(evaluate_binary(all_preds_flat, all_targets_flat))
    else:
        metrics.update(evaluate_continuous(all_preds_flat, all_targets_flat))

    # Per-dataset ranking metrics
    spearman_scores = []
    for pred, target in zip(all_preds, all_targets):
        if len(pred) >= 3 and np.std(target) > 0:
            r = evaluate_ranking(pred, target, top_k=max(1, int(np.sum(target > 0))))
            spearman_scores.append(r.get("spearman", 0.0))
    if spearman_scores:
        metrics["mean_spearman"] = float(np.mean(spearman_scores))

    return metrics


def evaluate_baselines(
    test_data: list[dict],
    label_mode: str,
) -> dict[str, dict]:
    """Evaluate baseline methods."""
    from tabpfn.interpretation.evaluation.metrics import evaluate_binary, evaluate_continuous

    output_mode = "binary" if label_mode in ("binary_direct", "binary_ancestry") else "continuous"

    # Random baseline
    random_preds = []
    targets_all = []
    for record in test_data:
        n_feat = record["feature_vectors"].shape[0]
        random_preds.append(np.random.random(n_feat))
        targets_all.append(record["labels"][label_mode])

    random_flat = np.concatenate(random_preds)
    targets_flat = np.concatenate(targets_all)

    baselines = {}
    if output_mode == "binary":
        baselines["random"] = evaluate_binary(random_flat, targets_flat)
    else:
        baselines["random"] = evaluate_continuous(random_flat, targets_flat)

    # Constant baseline (predict mean)
    mean_val = targets_flat.mean()
    const_preds = np.full_like(targets_flat, mean_val)
    if output_mode == "binary":
        baselines["constant_mean"] = evaluate_binary(const_preds, targets_flat)
    else:
        baselines["constant_mean"] = evaluate_continuous(const_preds, targets_flat)

    # Feature-vector norm baseline (use L2 norm of signal vector as importance)
    norm_preds = []
    for record in test_data:
        norms = np.linalg.norm(record["feature_vectors"], axis=1)
        norms = np.nan_to_num(norms, nan=0.0, posinf=0.0, neginf=0.0)
        if norms.max() > 0:
            norms = norms / norms.max()
        norm_preds.append(norms)
    norm_flat = np.concatenate(norm_preds)
    if output_mode == "binary":
        baselines["signal_norm"] = evaluate_binary(norm_flat, targets_flat)
    else:
        baselines["signal_norm"] = evaluate_continuous(norm_flat, targets_flat)

    return baselines


# ============================================================
# Phase 4: Full Experiment
# ============================================================

def run_full_experiment():
    """Run the complete experiment suite."""
    logger.info("=" * 60)
    logger.info("INTERPRETATION MODEL EXPERIMENT (v2.5 aligned)")
    logger.info("=" * 60)

    # ---- Load Pre-generated Data ----
    logger.info("\n[Phase 1] Loading pre-generated data from disk...")
    t_start = time.time()
    all_data = load_data()
    t_load = time.time() - t_start
    logger.info(f"Loaded {len(all_data)} datasets in {t_load:.1f}s")

    if len(all_data) < 30:
        logger.error(f"Only {len(all_data)} datasets available, need at least 30. "
                      "Run generate_interpretation_data.py first.")
        return

    # Split: 70% train, 15% val, 15% test
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    n_train = int(len(all_data) * 0.70)
    n_val = int(len(all_data) * 0.15)

    train_data = [all_data[i] for i in indices[:n_train]]
    val_data = [all_data[i] for i in indices[n_train:n_train + n_val]]
    test_data = [all_data[i] for i in indices[n_train + n_val:]]

    logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Analyze data stats
    feat_dims = [d["feature_vectors"].shape[1] for d in all_data]
    n_features_list = [d["feature_vectors"].shape[0] for d in all_data]
    n_parents = [np.sum(d["labels"]["binary_direct"] > 0) for d in all_data]
    n_ancestors = [np.sum(d["labels"]["binary_ancestry"] > 0) for d in all_data]

    data_stats = {
        "n_datasets": len(all_data),
        "feature_dim": int(np.mean(feat_dims)),
        "n_features_mean": float(np.mean(n_features_list)),
        "n_features_range": [int(np.min(n_features_list)), int(np.max(n_features_list))],
        "n_direct_parents_mean": float(np.mean(n_parents)),
        "n_ancestors_mean": float(np.mean(n_ancestors)),
        "parent_fraction_mean": float(np.mean([p/n for p, n in zip(n_parents, n_features_list)])),
        "ancestor_fraction_mean": float(np.mean([a/n for a, n in zip(n_ancestors, n_features_list)])),
        "generator": "tabpfn_prior (zzhang-cn/tabpfn-synthetic-data)",
    }
    logger.info(f"Data stats: {json.dumps(data_stats, indent=2)}")

    max_features = max(n_features_list) + 1
    input_dim = feat_dims[0]

    # ---- Main Experiments ----
    label_modes = ["binary_direct", "binary_ancestry", "graded_ancestry", "interventional"]
    model_variants = ["mlp", "transformer"]

    all_results = {}

    for label_mode in label_modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"[Phase 2] Label mode: {label_mode}")
        logger.info(f"{'='*60}")

        mode_results = {}

        # Baselines
        logger.info("  Evaluating baselines...")
        baselines = evaluate_baselines(test_data, label_mode)
        mode_results["baselines"] = baselines
        for name, metrics in baselines.items():
            logger.info(f"    {name}: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")

        # Train and evaluate each model variant
        for variant in model_variants:
            logger.info(f"\n  Training {variant} model for {label_mode}...")
            t0 = time.time()

            model, history = train_model(
                train_data, val_data,
                variant=variant,
                label_mode=label_mode,
                max_features=max_features,
                n_epochs=100,
                batch_size=16,
                lr=5e-4,
            )
            train_time = time.time() - t0

            # Save model
            model_path = RESULTS_DIR / f"model_{variant}_{label_mode}.pt"
            model.save(model_path)

            # Evaluate
            metrics = evaluate_model(model, test_data, label_mode, max_features)
            metrics["train_time_s"] = round(train_time, 1)
            metrics["best_train_loss"] = round(min(history["train_loss"]), 4)
            metrics["best_val_loss"] = round(min(history["val_loss"]), 4)
            metrics["n_epochs_trained"] = len(history["train_loss"])
            metrics["loss_history"] = {k: [round(v, 6) for v in vals] for k, vals in history.items()}

            mode_results[variant] = metrics
            logger.info(f"    {variant}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})}")

            del model
            torch.cuda.empty_cache()

        all_results[label_mode] = mode_results

    # ---- Feature Ablation Study ----
    logger.info("\n" + "=" * 60)
    logger.info("[Phase 3] Feature Ablation Study")
    logger.info("=" * 60)

    ablation_results = run_feature_ablation(
        train_data, val_data, test_data,
        max_features=max_features,
        label_modes=label_modes,
    )

    # ---- Save Results ----
    results_file = RESULTS_DIR / "experiment_results.json"
    full_results = {
        "data_stats": data_stats,
        "results": all_results,
        "ablation": ablation_results,
        "config": {
            "n_datasets": len(all_data),
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "max_features": max_features,
            "device": DEVICE,
            "data_load_time_s": round(t_load, 1),
            "generator": "tabpfn_prior",
        },
    }
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # ---- Print Summary ----
    print_summary(full_results)

    return full_results


# ============================================================
# Feature Ablation Study
# ============================================================

ABLATION_CONFIGS: dict[str, set[str] | None] = {
    "full": None,  # all categories (use pre-computed feature_vectors)
    "attention_only": {"between_features_attention", "between_items_attention"},
    "feat_attn_only": {"between_features_attention"},
    "item_attn_only": {"between_items_attention"},
    "embeddings_only": {"embeddings"},
    "mlp_only": {"mlp_activations"},
    "gradients_only": {"gradients"},
    "no_attention": {"embeddings", "mlp_activations", "gradients"},
    "no_embeddings": {"between_features_attention", "between_items_attention", "mlp_activations", "gradients"},
    "no_gradients": {"between_features_attention", "between_items_attention", "embeddings", "mlp_activations"},
}


def run_feature_ablation(
    train_data: list[dict],
    val_data: list[dict],
    test_data: list[dict],
    *,
    max_features: int,
    label_modes: list[str],
) -> dict:
    """Run the feature ablation study across signal category subsets.

    Uses pre-computed per-category feature vectors stored on disk.
    """
    ablation_results: dict[str, dict] = {}

    for config_name, categories in ABLATION_CONFIGS.items():
        logger.info(f"\n  Ablation: {config_name} (categories={categories or 'all'})")

        try:
            abl_train = _select_categories(train_data, categories)
            abl_val = _select_categories(val_data, categories)
            abl_test = _select_categories(test_data, categories)
        except Exception as e:
            logger.warning(f"    Ablation {config_name} failed: {e}")
            continue

        input_dim = abl_train[0]["feature_vectors"].shape[1]
        logger.info(f"    input_dim={input_dim}")

        config_results: dict[str, dict] = {"input_dim": input_dim}

        for label_mode in label_modes:
            logger.info(f"    Training MLP for {label_mode}...")
            try:
                model, history = train_model(
                    abl_train, abl_val,
                    variant="mlp",
                    label_mode=label_mode,
                    max_features=max_features,
                    n_epochs=80,
                    batch_size=16,
                    lr=5e-4,
                )
                metrics = evaluate_model(model, abl_test, label_mode, max_features)
                metrics["best_val_loss"] = round(min(history["val_loss"]), 4)
                metrics["loss_history"] = {k: [round(v, 6) for v in vals] for k, vals in history.items()}
                config_results[label_mode] = metrics
                logger.info(f"      {label_mode}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})}")

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"      {label_mode} failed: {e}")
                config_results[label_mode] = {"error": str(e)}

        ablation_results[config_name] = config_results

    return ablation_results


def _select_categories(
    data: list[dict],
    categories: set[str] | None,
) -> list[dict]:
    """Build feature vectors by concatenating selected per-category vectors.

    If *categories* is None, returns the original full feature_vectors.
    """
    if categories is None:
        return data

    result = []
    for record in data:
        cat_vecs = record.get("category_vectors", {})
        parts = []
        for cat in sorted(categories):
            if cat in cat_vecs:
                parts.append(cat_vecs[cat])
        if not parts:
            raise ValueError(f"No category vectors found for {categories}")
        fv = np.concatenate(parts, axis=1)
        result.append({
            "feature_vectors": fv,
            "labels": record["labels"],
            "metadata": record["metadata"],
        })
    return result


def print_summary(results: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)

    print(f"\nDataset Statistics:")
    ds = results["data_stats"]
    print(f"  Total datasets: {ds['n_datasets']}")
    print(f"  Features per dataset: {ds['n_features_mean']:.1f} (range: {ds['n_features_range']})")
    print(f"  Feature dim (D_total): {ds['feature_dim']}")
    print(f"  Direct parents per dataset: {ds['n_direct_parents_mean']:.1f} ({ds['parent_fraction_mean']*100:.1f}% of features)")
    print(f"  Total ancestors per dataset: {ds['n_ancestors_mean']:.1f} ({ds['ancestor_fraction_mean']*100:.1f}% of features)")
    print(f"  Generator: {ds.get('generator', 'unknown')}")

    for label_mode, mode_results in results["results"].items():
        is_binary = label_mode in ("binary_direct", "binary_ancestry")
        print(f"\n{'─' * 80}")
        print(f"Label Mode: {label_mode}")
        print(f"{'─' * 80}")

        if is_binary:
            header = f"{'Method':<25} {'AUROC':>8} {'AUPR':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}"
            print(header)
            print("─" * len(header))
            for method, metrics in mode_results.items():
                auroc = metrics.get("auroc", 0)
                aupr = metrics.get("aupr", 0)
                f1 = metrics.get("f1", 0)
                prec = metrics.get("precision", 0)
                rec = metrics.get("recall", 0)
                print(f"  {method:<23} {auroc:>8.4f} {aupr:>8.4f} {f1:>8.4f} {prec:>10.4f} {rec:>8.4f}")
        else:
            header = f"{'Method':<25} {'R²':>8} {'MAE':>8} {'Corr':>8} {'Spearman':>10}"
            print(header)
            print("─" * len(header))
            for method, metrics in mode_results.items():
                r2 = metrics.get("r2", 0)
                mae = metrics.get("mae", 0)
                corr = metrics.get("correlation", 0)
                spear = metrics.get("mean_spearman", metrics.get("spearman", 0))
                print(f"  {method:<23} {r2:>8.4f} {mae:>8.4f} {corr:>8.4f} {spear:>10.4f}")

    # Ablation summary
    ablation = results.get("ablation", {})
    if ablation:
        print(f"\n{'=' * 80}")
        print("FEATURE ABLATION STUDY")
        print(f"{'=' * 80}")

        # For binary_direct (AUROC)
        print(f"\n{'Config':<22} {'Dims':>5}  ", end="")
        label_modes = ["binary_direct", "binary_ancestry", "graded_ancestry", "interventional"]
        metric_names = ["AUROC", "AUROC", "R²", "R²"]
        for mn in metric_names:
            print(f"  {mn:>8}", end="")
        print()
        print("─" * 70)

        for config_name, config_data in ablation.items():
            dims = config_data.get("input_dim", "?")
            print(f"  {config_name:<20} {dims:>5}  ", end="")
            for lm in label_modes:
                m = config_data.get(lm, {})
                if isinstance(m, dict) and "error" not in m:
                    if lm in ("binary_direct", "binary_ancestry"):
                        val = m.get("auroc", 0)
                    else:
                        val = m.get("r2", 0)
                    print(f"  {val:>8.4f}", end="")
                else:
                    print(f"  {'N/A':>8}", end="")
            print()

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_full_experiment()
