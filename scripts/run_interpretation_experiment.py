#!/usr/bin/env python
"""Comprehensive experiment: Train and evaluate the interpretation model.

This script:
1. Generates synthetic datasets from SCMs with known causal structure
2. Fits TabPFN on each dataset and extracts internal signals
3. Trains interpretation models (MLP and Transformer variants)
4. Evaluates across all 4 label modes
5. Compares against baselines
6. Produces a comprehensive performance report
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/interpretation_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ============================================================
# Phase 1: Generate Training Data
# ============================================================

def generate_training_data(
    n_datasets: int = 200,
    n_features_range: tuple[int, int] = (5, 25),
    n_samples_range: tuple[int, int] = (80, 300),
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic datasets, run TabPFN, extract signals."""
    from tabpfn import TabPFNRegressor
    from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
    from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
    from tabpfn.interpretation.synthetic_data.label_generator import compute_all_labels
    from tabpfn.interpretation.synthetic_data.scm_generator import SCMGenerator

    gen = SCMGenerator(
        n_features_range=n_features_range,
        n_samples_range=n_samples_range,
        seed=seed,
    )
    extractor = SignalExtractor(extract_gradients=False)
    processor = SignalProcessor()

    all_data = []
    timings = {"scm": [], "tabpfn_fit": [], "extract": [], "process": []}

    for i in range(n_datasets):
        try:
            # Step 1: Generate SCM dataset
            t0 = time.time()
            dataset = gen.generate()
            timings["scm"].append(time.time() - t0)

            n_total = dataset.X.shape[0]
            n_train = max(int(n_total * 0.7), 20)
            X_train = dataset.X[:n_train].astype(np.float32)
            X_test = dataset.X[n_train:].astype(np.float32)
            y_train = dataset.y[:n_train].astype(np.float32)

            # Step 2: Fit TabPFN
            t0 = time.time()
            reg = TabPFNRegressor(n_estimators=1, device=DEVICE)
            reg.fit(X_train, y_train)
            timings["tabpfn_fit"].append(time.time() - t0)

            # Step 3: Extract signals
            t0 = time.time()
            signals = extractor.extract(reg, X_train, y_train, X_test)
            timings["extract"].append(time.time() - t0)

            # Step 4: Process to feature vectors
            t0 = time.time()
            feature_vectors = processor.process(signals)
            timings["process"].append(time.time() - t0)

            # Step 5: Compute labels
            labels = compute_all_labels(dataset)

            all_data.append({
                "feature_vectors": feature_vectors,
                "labels": labels,
                "metadata": dataset.metadata,
            })

            if (i + 1) % 20 == 0:
                logger.info(
                    f"Generated {i+1}/{n_datasets} datasets. "
                    f"Last: n_feat={dataset.X.shape[1]}, "
                    f"n_parents={dataset.metadata['n_target_parents']}, "
                    f"feat_dim={feature_vectors.shape[1]}"
                )

            # Free GPU memory
            del reg, signals
            torch.cuda.empty_cache()

        except Exception as e:
            logger.warning(f"Dataset {i+1} failed: {e}")
            continue

    logger.info(f"Successfully generated {len(all_data)}/{n_datasets} datasets")
    logger.info(f"Timings (mean): SCM={np.mean(timings['scm']):.3f}s, "
                f"TabPFN={np.mean(timings['tabpfn_fit']):.3f}s, "
                f"Extract={np.mean(timings['extract']):.3f}s, "
                f"Process={np.mean(timings['process']):.3f}s")

    return all_data


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
    logger.info("INTERPRETATION MODEL EXPERIMENT")
    logger.info("=" * 60)

    # ---- Data Generation ----
    logger.info("\n[Phase 1] Generating training data...")
    t_start = time.time()
    all_data = generate_training_data(
        n_datasets=200,
        n_features_range=(5, 25),
        n_samples_range=(80, 300),
        seed=42,
    )
    t_datagen = time.time() - t_start
    logger.info(f"Data generation took {t_datagen:.1f}s")

    if len(all_data) < 20:
        logger.error(f"Only {len(all_data)} datasets generated, need at least 20")
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
    }
    logger.info(f"Data stats: {json.dumps(data_stats, indent=2)}")

    max_features = max(n_features_list) + 1
    input_dim = feat_dims[0]

    # ---- Experiments ----
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

            mode_results[variant] = metrics
            logger.info(f"    {variant}: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})}")

            # Free GPU
            del model
            torch.cuda.empty_cache()

        all_results[label_mode] = mode_results

    # ---- Save Results ----
    results_file = RESULTS_DIR / "experiment_results.json"
    full_results = {
        "data_stats": data_stats,
        "results": all_results,
        "config": {
            "n_datasets": len(all_data),
            "n_train": len(train_data),
            "n_val": len(val_data),
            "n_test": len(test_data),
            "max_features": max_features,
            "device": DEVICE,
            "data_generation_time_s": round(t_datagen, 1),
        },
    }
    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)
    logger.info(f"\nResults saved to {results_file}")

    # ---- Print Summary ----
    print_summary(full_results)

    return full_results


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

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_full_experiment()
