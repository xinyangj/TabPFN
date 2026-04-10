#!/usr/bin/env python
"""Retrain the best v2 interpretation model and save checkpoint with full config.

Trains deep_lr1e4 ([1024,512,256,128], lr=1e-4) on v2 cache with 691-dim
features (attn + items_attn + mlp + gradients). Saves model weights, config,
and global normalization stats for downstream GRN evaluation.

Usage:
    python scripts/retrain_best_v2.py [--device cuda:1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------- Configuration (best v2 config) ----------
CONFIG = {
    "hidden_dims": [1024, 512, 256, 128],
    "lr": 1e-4,
    "dropout": 0.1,
    "norm": "layer",
    "activation": "gelu",
    "batch_size": 512,
    "max_epochs": 200,
    "patience": 25,
    "weight_decay": 1e-4,
    "label_mode": "binary_direct",
    "input_categories": sorted([
        "between_features_attention",
        "between_items_attention",
        "mlp_activations",
        "gradients",
    ]),
    "expected_dim": 691,
}

CACHE_DIR = Path("data/interpretation_cache_v2")
OUT_PATH = Path("results/interpretation_experiments/best_v2_model.pt")


class FastDataset(Dataset):
    def __init__(self, data, label_mode, max_features, augment=False):
        n = len(data)
        d = data[0]["feature_vectors"].shape[1]
        self.augment = augment
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
            return {"features": features, "labels": labels, "mask": self.masks[idx]}
        return {"features": self.features[idx], "labels": self.labels[idx], "mask": self.masks[idx]}


def load_and_split_data():
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


def select_categories(data, categories, expected_dim=None):
    result = []
    skipped = 0
    for record in data:
        cat_vecs = record.get("category_vectors", {})
        parts = [cat_vecs[c] for c in sorted(categories) if c in cat_vecs]
        if not parts:
            skipped += 1
            continue
        fv = np.concatenate(parts, axis=1)
        if expected_dim is not None and fv.shape[1] != expected_dim:
            skipped += 1
            continue
        result.append({
            "feature_vectors": fv,
            "labels": record["labels"],
            "metadata": record["metadata"],
        })
    if skipped:
        logger.info(f"  Skipped {skipped} datasets with missing/mismatched categories")
    return result


def compute_global_norm_stats(dataset):
    """Compute mean/std across all valid feature positions in the training set."""
    all_features = []
    for i in range(len(dataset)):
        item = dataset[i]
        mask = item["mask"]
        features = item["features"]
        nf = int(mask.sum().item())
        if nf > 0:
            all_features.append(features[:nf])
    all_cat = torch.cat(all_features, dim=0)
    return all_cat.mean(dim=0), all_cat.std(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading v2 cache from {CACHE_DIR}...")
    train_data, val_data, test_data = load_and_split_data()

    categories = set(CONFIG["input_categories"])
    train_sel = select_categories(train_data, categories, expected_dim=CONFIG["expected_dim"])
    val_sel = select_categories(val_data, categories, expected_dim=CONFIG["expected_dim"])
    test_sel = select_categories(test_data, categories, expected_dim=CONFIG["expected_dim"])
    logger.info(f"After selection: {len(train_sel)} train, {len(val_sel)} val, {len(test_sel)} test")

    input_dim = train_sel[0]["feature_vectors"].shape[1]
    max_features = max(
        r["feature_vectors"].shape[0] for r in train_sel + val_sel + test_sel
    ) + 1
    logger.info(f"Input dim: {input_dim}, max_features: {max_features}")

    # Build datasets
    train_ds = FastDataset(train_sel, CONFIG["label_mode"], max_features, augment=True)
    val_ds = FastDataset(val_sel, CONFIG["label_mode"], max_features, augment=False)
    test_ds = FastDataset(test_sel, CONFIG["label_mode"], max_features, augment=False)

    del train_data, val_data, test_data, train_sel, val_sel, test_sel

    # Compute global normalization stats
    logger.info("Computing global normalization stats from training data...")
    global_mean, global_std = compute_global_norm_stats(train_ds)
    logger.info(f"Global mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")

    # Create model
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.evaluation.metrics import evaluate_binary, evaluate_ranking

    model = InterpretationModel(
        "mlp", input_dim=input_dim, output_mode="binary",
        hidden_dims=CONFIG["hidden_dims"], dropout=CONFIG["dropout"],
        norm=CONFIG["norm"], activation=CONFIG["activation"],
    ).to(device)
    # Note: no global norm stats — matching the tuning experiment setup

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    loss_fn = InterpretationLoss(mode="binary").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG["max_epochs"])

    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True, num_workers=2)

    # Train
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(CONFIG["max_epochs"]):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            preds = model(features, mask=mask)
            loss = loss_fn(preds, labels, mask=mask.bool())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                preds = model(features, mask=mask)
                loss = loss_fn(preds, labels, mask=mask.bool())
                val_loss_sum += loss.item()
                val_batches += 1
        val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step()

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_epoch = epoch
            marker = " *"
        else:
            no_improve += 1

        if epoch % 10 == 0 or marker:
            elapsed = time.time() - t0
            logger.info(f"  epoch {epoch:3d} train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} ({elapsed:.0f}s){marker}")

        if no_improve >= CONFIG["patience"]:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Evaluate on test set
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=CONFIG["batch_size"], shuffle=False, pin_memory=True)
    all_preds, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            preds = model(features, mask=mask)
            all_preds.append(preds.cpu())
            all_labels.append(batch["labels"])
            all_masks.append(batch["mask"])

    preds_cat = torch.cat(all_preds)
    labels_cat = torch.cat(all_labels)
    masks_cat = torch.cat(all_masks)

    valid = masks_cat.bool().flatten()
    pred_flat = torch.sigmoid(preds_cat).flatten()[valid].numpy()
    label_flat = labels_cat.flatten()[valid].numpy()

    metrics_bin = evaluate_binary(pred_flat, label_flat)
    metrics_rank = evaluate_ranking(pred_flat, label_flat)
    total_time = time.time() - t0

    logger.info(f"Test AUROC: {metrics_bin['auroc']:.4f}")
    logger.info(f"Test AUPR:  {metrics_bin['aupr']:.4f}")
    logger.info(f"Test Spearman: {metrics_rank.get('spearman', 0):.4f}")
    logger.info(f"Best epoch: {best_epoch}, Total time: {total_time:.0f}s")

    # Save checkpoint with full config and norm stats
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "variant": "mlp",
        "input_dim": input_dim,
        "output_mode": "binary",
        "config": CONFIG,
        "hidden_dims": CONFIG["hidden_dims"],
        "dropout": CONFIG["dropout"],
        "norm": CONFIG["norm"],
        "activation": CONFIG["activation"],
        "state_dict": best_state,
        "global_mean": global_mean,
        "global_std": global_std,
        "test_metrics": {
            "auroc": metrics_bin.get("auroc", 0),
            "aupr": metrics_bin.get("aupr", 0),
            "f1": metrics_bin.get("f1", 0),
            "spearman": metrics_rank.get("spearman", 0),
        },
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time_s": total_time,
    }
    torch.save(checkpoint, OUT_PATH)
    logger.info(f"Checkpoint saved to {OUT_PATH}")

    # Also save a human-readable summary
    summary = {
        "config": CONFIG,
        "input_dim": input_dim,
        "n_params": n_params,
        "test_metrics": checkpoint["test_metrics"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time_s": total_time,
    }
    summary_path = OUT_PATH.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
