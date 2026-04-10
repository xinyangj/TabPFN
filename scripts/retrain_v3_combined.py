#!/usr/bin/env python
"""Retrain interpretation model on combined v2 + v3 caches.

Loads v2 cache (148K datasets, 3-30 features) and v3 cache (high-feature
supplement, 10-150 features), combines them, and trains with the best
configuration from tuning (narrow_lr1e4: 512×256×128×64, lr=1e-4).

Usage:
    python scripts/retrain_v3_combined.py --device cuda:1
    python scripts/retrain_v3_combined.py --device cuda:1 --config narrow  # narrow arch
    python scripts/retrain_v3_combined.py --device cuda:1 --config deep    # deep arch
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

# ---------- Configurations ----------
CONFIGS = {
    "narrow": {
        "hidden_dims": [512, 256, 128, 64],
        "lr": 1e-4,
        "dropout": 0.1,
        "norm": "layer",
        "activation": "gelu",
        "batch_size": 512,
        "max_epochs": 250,
        "patience": 25,
        "weight_decay": 1e-4,
    },
    "deep": {
        "hidden_dims": [1024, 512, 256, 128],
        "lr": 1e-4,
        "dropout": 0.1,
        "norm": "layer",
        "activation": "gelu",
        "batch_size": 512,
        "max_epochs": 200,
        "patience": 25,
        "weight_decay": 1e-4,
    },
}

LABEL_MODE = "binary_direct"
INPUT_CATEGORIES = sorted([
    "between_features_attention",
    "between_items_attention",
    "mlp_activations",
    "gradients",
])
EXPECTED_DIM = 691

V2_CACHE = Path("data/interpretation_cache_v2")
V3_CACHE = Path("data/interpretation_cache_v3")
OUT_DIR = Path("results/interpretation_experiments")


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


def load_cache(cache_dir: Path) -> list[dict]:
    """Load all datasets from a cache directory."""
    from scripts.generate_interpretation_data import load_dataset
    npz_files = sorted(cache_dir.glob("dataset_*.npz"))
    logger.info(f"Loading {len(npz_files)} datasets from {cache_dir}")
    datasets = []
    for p in npz_files:
        try:
            datasets.append(load_dataset(p))
        except Exception as e:
            logger.warning(f"Failed to load {p.name}: {e}")
    logger.info(f"Loaded {len(datasets)} datasets")
    return datasets


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
    parser.add_argument("--config", default="narrow", choices=["narrow", "deep"])
    parser.add_argument("--v2_cache", default=str(V2_CACHE))
    parser.add_argument("--v3_cache", default=str(V3_CACHE))
    parser.add_argument("--output", default=None, help="Output checkpoint path")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    config = CONFIGS[args.config]
    logger.info(f"Using device: {device}, config: {args.config}")
    logger.info(f"Architecture: {config['hidden_dims']}, LR: {config['lr']}")

    v2_dir = Path(args.v2_cache)
    v3_dir = Path(args.v3_cache)

    out_path = Path(args.output) if args.output else OUT_DIR / f"best_v3_{args.config}_model.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load both caches
    logger.info("Loading v2 cache...")
    v2_data = load_cache(v2_dir)
    logger.info("Loading v3 cache...")
    v3_data = load_cache(v3_dir)
    logger.info(f"Combined: {len(v2_data)} v2 + {len(v3_data)} v3 = {len(v2_data)+len(v3_data)} total")

    # Combine and shuffle
    all_data = v2_data + v3_data
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_data))
    all_data = [all_data[i] for i in indices]

    # Split: 70/15/15
    n = len(all_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]
    logger.info(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    del all_data

    # Select categories
    categories = set(INPUT_CATEGORIES)
    train_sel = select_categories(train_data, categories, expected_dim=EXPECTED_DIM)
    val_sel = select_categories(val_data, categories, expected_dim=EXPECTED_DIM)
    test_sel = select_categories(test_data, categories, expected_dim=EXPECTED_DIM)
    logger.info(f"After selection: {len(train_sel)} train, {len(val_sel)} val, {len(test_sel)} test")

    # Feature count stats
    fc_train = [r["feature_vectors"].shape[0] for r in train_sel]
    logger.info(f"Feature count stats: mean={np.mean(fc_train):.1f}, max={max(fc_train)}, "
                f">=50: {sum(1 for x in fc_train if x >= 50)}, >=100: {sum(1 for x in fc_train if x >= 100)}")

    input_dim = train_sel[0]["feature_vectors"].shape[1]
    max_features = max(
        r["feature_vectors"].shape[0] for r in train_sel + val_sel + test_sel
    ) + 1
    logger.info(f"Input dim: {input_dim}, max_features: {max_features}")

    # Build datasets
    train_ds = FastDataset(train_sel, LABEL_MODE, max_features, augment=True)
    val_ds = FastDataset(val_sel, LABEL_MODE, max_features, augment=False)
    test_ds = FastDataset(test_sel, LABEL_MODE, max_features, augment=False)
    del train_data, val_data, test_data, train_sel, val_sel, test_sel

    # Compute global normalization stats (saved but not applied to model)
    logger.info("Computing global normalization stats...")
    global_mean, global_std = compute_global_norm_stats(train_ds)
    logger.info(f"Global mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")

    # Create model
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.evaluation.metrics import evaluate_binary, evaluate_ranking

    model = InterpretationModel(
        "mlp", input_dim=input_dim, output_mode="binary",
        hidden_dims=config["hidden_dims"], dropout=config["dropout"],
        norm=config["norm"], activation=config["activation"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    loss_fn = InterpretationLoss(mode="binary").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["max_epochs"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
                              drop_last=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
                            pin_memory=True, num_workers=2)

    # Train
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(config["max_epochs"]):
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
            logger.info(f"  epoch {epoch:3d} train={train_loss:.4f} val={val_loss:.4f} "
                        f"best={best_val_loss:.4f} ({elapsed:.0f}s){marker}")

        if no_improve >= config["patience"]:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Evaluate on test set
    if best_state:
        model.load_state_dict(best_state)
        model.to(device)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False, pin_memory=True)
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

    # Save checkpoint
    checkpoint = {
        "variant": "mlp",
        "input_dim": input_dim,
        "output_mode": "binary",
        "config": {
            **config,
            "label_mode": LABEL_MODE,
            "input_categories": INPUT_CATEGORIES,
            "expected_dim": EXPECTED_DIM,
        },
        "hidden_dims": config["hidden_dims"],
        "dropout": config["dropout"],
        "norm": config["norm"],
        "activation": config["activation"],
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
        "data_sources": {
            "v2_cache": str(v2_dir),
            "v3_cache": str(v3_dir),
            "n_v2": len(v2_data) if 'v2_data' in dir() else "unknown",
            "n_v3": len(v3_data) if 'v3_data' in dir() else "unknown",
        },
    }
    torch.save(checkpoint, out_path)
    logger.info(f"Checkpoint saved to {out_path}")

    summary = {
        "config_name": args.config,
        "config": {**config, "label_mode": LABEL_MODE},
        "input_dim": input_dim,
        "max_features": max_features,
        "n_params": n_params,
        "test_metrics": checkpoint["test_metrics"],
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "total_time_s": total_time,
    }
    summary_path = out_path.with_suffix(".json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
