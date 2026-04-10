#!/usr/bin/env python
"""MLP Hyperparameter Tuning for the Interpretation Model.

Runs a grid search over learning rate and architecture configurations,
training on binary_direct with no_embeddings input (579 dims).

Optimized for speed:
- Pre-tensorizes all data at startup (eliminates per-batch numpy→torch conversion)
- Large batch size (512) to saturate GPU
- pin_memory for async host→device transfer

Usage:
    python scripts/tune_mlp_hyperparams.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s   %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results/interpretation_experiments")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path("data/interpretation_cache")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Ensure project root is on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ── Grid definition ──────────────────────────────────────────────────

TUNING_CONFIGS = [
    {
        "name": "baseline_lr1e-4",
        "hidden_dims": [512, 256, 128],
        "lr": 1e-4,
    },
    {
        "name": "baseline_lr5e-4",
        "hidden_dims": [512, 256, 128],
        "lr": 5e-4,
    },
    {
        "name": "deeper_lr1e-4",
        "hidden_dims": [1024, 512, 256, 128],
        "lr": 1e-4,
    },
    {
        "name": "deeper_lr5e-4",
        "hidden_dims": [1024, 512, 256, 128],
        "lr": 5e-4,
    },
]

# Fixed hyperparameters
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 25
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
LABEL_MODE = "binary_direct"
INPUT_CATEGORIES = {"between_features_attention", "between_items_attention", "mlp_activations"}


# ── Fast pre-tensorized dataset ──────────────────────────────────────

class FastDataset(Dataset):
    """Pre-tensorized dataset — __getitem__ is pure tensor indexing."""

    def __init__(
        self,
        data: list[dict],
        label_mode: str,
        max_features: int,
        augment: bool = False,
    ):
        n = len(data)
        d = data[0]["feature_vectors"].shape[1]
        self.augment = augment

        # Pre-allocate contiguous tensors
        self.features = torch.zeros(n, max_features, d)
        self.labels = torch.zeros(n, max_features)
        self.masks = torch.zeros(n, max_features)
        self.n_features = torch.zeros(n, dtype=torch.long)

        for i, record in enumerate(data):
            fv = record["feature_vectors"]
            lab = record["labels"][label_mode]
            nf = min(fv.shape[0], max_features)
            self.features[i, :nf] = torch.from_numpy(fv[:nf].astype(np.float32))
            self.labels[i, :nf] = torch.from_numpy(lab[:nf].astype(np.float32))
            self.masks[i, :nf] = 1.0
            self.n_features[i] = nf

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.augment:
            nf = int(self.n_features[idx])
            perm = torch.randperm(nf)
            features = self.features[idx].clone()
            labels = self.labels[idx].clone()
            features[:nf] = features[perm]
            labels[:nf] = labels[perm]
            return {
                "features": features,
                "labels": labels,
                "mask": self.masks[idx],
                "n_features": self.n_features[idx],
            }
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "mask": self.masks[idx],
            "n_features": self.n_features[idx],
        }


# ── Data loading ─────────────────────────────────────────────────────

def load_and_split_data():
    """Load data from cache and split into train/val/test."""
    from scripts.generate_interpretation_data import load_all_datasets

    all_data = load_all_datasets(CACHE_DIR)
    n = len(all_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    return train_data, val_data, test_data


def select_categories(data: list[dict], categories: set[str]) -> list[dict]:
    """Build feature vectors from selected signal categories."""
    result = []
    for record in data:
        cat_vecs = record.get("category_vectors", {})
        parts = [cat_vecs[c] for c in sorted(categories) if c in cat_vecs]
        if not parts:
            raise ValueError(f"No category vectors for {categories}")
        fv = np.concatenate(parts, axis=1)
        result.append({
            "feature_vectors": fv,
            "labels": record["labels"],
            "metadata": record["metadata"],
        })
    return result


# ── Training ─────────────────────────────────────────────────────────

def train_and_evaluate(
    train_ds: FastDataset,
    val_ds: FastDataset,
    test_ds: FastDataset,
    *,
    hidden_dims: list[int],
    lr: float,
) -> dict:
    """Train MLP with given config and return metrics + history."""
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.evaluation.metrics import (
        evaluate_binary,
        evaluate_ranking,
    )

    input_dim = train_ds.features.shape[2]

    model = InterpretationModel(
        "mlp", input_dim=input_dim, output_mode="binary",
        hidden_dims=hidden_dims, dropout=DROPOUT,
    ).to(DEVICE)

    loss_fn = InterpretationLoss(mode="binary").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        pin_memory=True, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=0,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    Model params: {n_params:,}")
    logger.info(f"    Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    t0 = time.time()
    for epoch in range(MAX_EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
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
                features = batch["features"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].to(DEVICE, non_blocking=True)
                mask = batch["mask"].to(DEVICE, non_blocking=True)
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
            elapsed = time.time() - t0
            logger.info(
                f"    Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}"
                f"  (best_val={best_val_loss:.4f}, no_improve={no_improve})"
                f"  [{elapsed:.0f}s elapsed]"
            )

        if no_improve >= PATIENCE:
            logger.info(f"    Early stopping at epoch {epoch+1}")
            break

        # Check for NaN
        if not np.isfinite(train_loss):
            logger.warning(f"    NaN loss at epoch {epoch+1}, stopping")
            break

    train_time = time.time() - t0

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    # Evaluate on test set (per-dataset)
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            batch = test_ds[i]
            features = batch["features"].unsqueeze(0).to(DEVICE)
            mask = batch["mask"].unsqueeze(0).to(DEVICE)
            labels = batch["labels"].numpy()

            preds = torch.sigmoid(model(features, mask=mask))
            preds = preds.cpu().numpy().squeeze(0)

            n_feat = int(batch["n_features"])
            all_preds.append(preds[:n_feat])
            all_targets.append(labels[:n_feat])

    all_preds_flat = np.concatenate(all_preds)
    all_targets_flat = np.concatenate(all_targets)

    metrics = evaluate_binary(all_preds_flat, all_targets_flat)

    # Per-dataset Spearman
    spearman_scores = []
    for pred, target in zip(all_preds, all_targets):
        if len(pred) >= 3 and np.std(target) > 0:
            r = evaluate_ranking(pred, target, top_k=max(1, int(np.sum(target > 0))))
            spearman_scores.append(r.get("spearman", 0.0))
    if spearman_scores:
        metrics["mean_spearman"] = float(np.mean(spearman_scores))
    metrics["train_time_s"] = round(train_time, 1)
    metrics["n_epochs_trained"] = len(history["train_loss"])
    metrics["best_val_loss"] = round(best_val_loss, 6)
    metrics["best_train_loss"] = round(min(history["train_loss"]), 6)
    metrics["n_params"] = n_params
    metrics["loss_history"] = {
        k: [round(v, 6) for v in vals] for k, vals in history.items()
    }

    del model
    torch.cuda.empty_cache()

    return metrics


# ── Main ─────────────────────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("MLP Hyperparameter Tuning — Phase 1")
    logger.info("=" * 60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Input categories: {INPUT_CATEGORIES}")
    logger.info(f"Label mode: {LABEL_MODE}")
    logger.info(f"Batch size: {BATCH_SIZE}, Max epochs: {MAX_EPOCHS}, Patience: {PATIENCE}")
    logger.info(f"Configs to run: {len(TUNING_CONFIGS)}")

    # Load data
    logger.info("\n[Phase 1] Loading data...")
    train_data, val_data, test_data = load_and_split_data()

    # Select categories (no_embeddings)
    logger.info("Selecting categories (no_embeddings)...")
    train_sel = select_categories(train_data, INPUT_CATEGORIES)
    val_sel = select_categories(val_data, INPUT_CATEGORIES)
    test_sel = select_categories(test_data, INPUT_CATEGORIES)

    input_dim = train_sel[0]["feature_vectors"].shape[1]
    max_features = max(r["feature_vectors"].shape[0] for r in train_sel + val_sel + test_sel) + 1
    logger.info(f"Input dim: {input_dim}, Max features: {max_features}")

    # Free raw data
    del train_data, val_data, test_data

    # Pre-tensorize datasets (one-time cost, eliminates per-batch numpy ops)
    logger.info("Pre-tensorizing datasets...")
    t_tensor = time.time()
    train_ds = FastDataset(train_sel, LABEL_MODE, max_features, augment=True)
    val_ds = FastDataset(val_sel, LABEL_MODE, max_features, augment=False)
    test_ds = FastDataset(test_sel, LABEL_MODE, max_features, augment=False)
    del train_sel, val_sel, test_sel
    logger.info(
        f"Pre-tensorized in {time.time() - t_tensor:.1f}s — "
        f"train: {train_ds.features.shape}, val: {val_ds.features.shape}, test: {test_ds.features.shape}"
    )

    results = {}
    t_total = time.time()

    for i, cfg in enumerate(TUNING_CONFIGS):
        name = cfg["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(TUNING_CONFIGS)}] Config: {name}")
        logger.info(f"  hidden_dims={cfg['hidden_dims']}, lr={cfg['lr']}")
        logger.info(f"{'='*60}")

        try:
            metrics = train_and_evaluate(
                train_ds, val_ds, test_ds,
                hidden_dims=cfg["hidden_dims"],
                lr=cfg["lr"],
            )

            results[name] = {
                "config": cfg,
                "metrics": metrics,
            }

            auroc = metrics.get("auroc", 0)
            aupr = metrics.get("aupr", 0)
            logger.info(
                f"  Result: AUROC={auroc:.4f}, AUPR={aupr:.4f}, "
                f"val_loss={metrics['best_val_loss']:.4f}, "
                f"epochs={metrics['n_epochs_trained']}, "
                f"time={metrics['train_time_s']:.0f}s"
            )

        except Exception as e:
            logger.error(f"  Config {name} FAILED: {e}", exc_info=True)
            results[name] = {"config": cfg, "error": str(e)}

    total_time = time.time() - t_total

    # Save results
    output_path = RESULTS_DIR / "tuning_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TUNING RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Config':<25s} {'AUROC':>8s} {'AUPR':>8s} {'ValLoss':>10s} {'Epochs':>7s} {'Time':>7s}")
    logger.info("-" * 70)

    ranked = []
    for name, r in results.items():
        if "error" in r:
            logger.info(f"  {name:<25s} FAILED: {r['error']}")
            continue
        m = r["metrics"]
        auroc = m.get("auroc", 0)
        aupr = m.get("aupr", 0)
        ranked.append((auroc, name, m))

    for auroc, name, m in sorted(ranked, reverse=True):
        logger.info(
            f"  {name:<25s} {m.get('auroc', 0):>8.4f} {m.get('aupr', 0):>8.4f} "
            f"{m['best_val_loss']:>10.4f} {m['n_epochs_trained']:>7d} "
            f"{m['train_time_s']:>6.0f}s"
        )

    if ranked:
        best_auroc, best_name, best_m = max(ranked)
        logger.info(f"\n🏆 Best config: {best_name} (AUROC={best_auroc:.4f})")

    logger.info(f"\nTotal time: {total_time/3600:.1f} hours")


if __name__ == "__main__":
    main()
