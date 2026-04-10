#!/usr/bin/env python
"""Run LR × architecture tuning on v2 cache with attn + gradient features (691 dims).

Grid: 3 LR × 3 architectures = 9 configs.
All use GELU, LayerNorm, batch_size=512, patience=25.

Usage:
    python scripts/run_v2_tuning.py
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

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path("data/interpretation_cache_v2")

BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 25
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
LABEL_MODE = "binary_direct"

INPUT_CATEGORIES = {
    "between_features_attention",
    "between_items_attention",
    "mlp_activations",
    "gradients",
}

CONFIGS = [
    # (name, hidden_dims, lr)
    ("deep_lr5e5",   [1024, 512, 256, 128], 5e-5),
    ("deep_lr1e4",   [1024, 512, 256, 128], 1e-4),
    ("deep_lr3e4",   [1024, 512, 256, 128], 3e-4),
    ("wide_lr5e5",   [2048, 1024, 512],     5e-5),
    ("wide_lr1e4",   [2048, 1024, 512],     1e-4),
    ("wide_lr3e4",   [2048, 1024, 512],     3e-4),
    ("narrow_lr5e5", [512, 256, 128, 64],   5e-5),
    ("narrow_lr1e4", [512, 256, 128, 64],   1e-4),
    ("narrow_lr3e4", [512, 256, 128, 64],   3e-4),
]


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


def train_config(name, hidden_dims, lr, train_ds, val_ds, test_ds, input_dim):
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.evaluation.metrics import evaluate_binary, evaluate_ranking

    model = InterpretationModel(
        "mlp", input_dim=input_dim, output_mode="binary",
        hidden_dims=hidden_dims, dropout=DROPOUT,
        norm="layer", activation="gelu",
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{name}] hidden={hidden_dims}, lr={lr}, input_dim={input_dim}, params={n_params:,}")

    loss_fn = InterpretationLoss(mode="binary").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        drop_last=True, pin_memory=True, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        pin_memory=True, num_workers=2,
    )

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    best_epoch = 0
    t0 = time.time()

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)
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
                features = batch["features"].to(DEVICE, non_blocking=True)
                labels = batch["labels"].to(DEVICE, non_blocking=True)
                mask = batch["mask"].to(DEVICE, non_blocking=True)
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
            logger.info(
                f"  [{name}] epoch {epoch:3d} train={train_loss:.4f} "
                f"val={val_loss:.4f} best={best_val_loss:.4f} ({elapsed:.0f}s){marker}"
            )

        if no_improve >= PATIENCE:
            logger.info(f"  [{name}] Early stopping at epoch {epoch}")
            break

    # Evaluate on test set
    if best_state:
        model.load_state_dict(best_state)
        model.to(DEVICE)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    all_preds, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(DEVICE, non_blocking=True)
            mask = batch["mask"].to(DEVICE, non_blocking=True)
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

    # Free GPU memory
    del model, loss_fn, optimizer, scheduler
    torch.cuda.empty_cache()

    return {
        "config": name,
        "hidden_dims": hidden_dims,
        "lr": lr,
        "input_dim": input_dim,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_train_loss": train_loss,
        "auroc": metrics_bin.get("auroc", 0),
        "aupr": metrics_bin.get("aupr", 0),
        "f1": metrics_bin.get("f1", 0),
        "precision": metrics_bin.get("precision", 0),
        "recall": metrics_bin.get("recall", 0),
        "spearman": metrics_rank.get("spearman", 0) if metrics_rank else 0,
        "total_time_s": total_time,
    }


def main():
    logger.info(f"Loading v2 cache from {CACHE_DIR}...")
    train_data, val_data, test_data = load_and_split_data()

    logger.info(f"Selecting categories: {sorted(INPUT_CATEGORIES)}")
    train_sel = select_categories(train_data, INPUT_CATEGORIES, expected_dim=691)
    val_sel = select_categories(val_data, INPUT_CATEGORIES, expected_dim=691)
    test_sel = select_categories(test_data, INPUT_CATEGORIES, expected_dim=691)
    logger.info(f"After selection: {len(train_sel)} train, {len(val_sel)} val, {len(test_sel)} test")

    input_dim = train_sel[0]["feature_vectors"].shape[1]
    max_features = max(
        r["feature_vectors"].shape[0] for r in train_sel + val_sel + test_sel
    ) + 1
    logger.info(f"Input dim: {input_dim}, max_features: {max_features}")

    # Build datasets once — shared across all configs
    train_ds = FastDataset(train_sel, LABEL_MODE, max_features, augment=True)
    val_ds = FastDataset(val_sel, LABEL_MODE, max_features, augment=False)
    test_ds = FastDataset(test_sel, LABEL_MODE, max_features, augment=False)
    logger.info(f"Datasets tensorized: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Free raw data to save memory
    del train_data, val_data, test_data, train_sel, val_sel, test_sel

    results = []
    for name, hidden_dims, lr in CONFIGS:
        r = train_config(name, hidden_dims, lr, train_ds, val_ds, test_ds, input_dim)
        results.append(r)
        logger.info(
            f"[{name}] AUROC={r['auroc']:.4f} AUPR={r['aupr']:.4f} "
            f"Spearman={r['spearman']:.4f} Time={r['total_time_s']/60:.1f}m"
        )

        # Save intermediate results after each config
        out_path = Path("results/interpretation_experiments/v2_tuning_results.json")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n{'='*100}")
    print(f"V2 TUNING RESULTS (148K dataset, binary_direct, {input_dim} dims = attn+items+mlp+grad)")
    print(f"{'='*100}")
    print(
        f"{'Config':<18s} {'Arch':<24s} {'LR':>7s} {'AUROC':>8s} "
        f"{'AUPR':>8s} {'Spearman':>9s} {'ValLoss':>8s} {'Epoch':>6s} {'Time':>8s}"
    )
    print("-" * 100)

    best_auroc = max(r["auroc"] for r in results)
    for r in sorted(results, key=lambda x: x["auroc"], reverse=True):
        tag = " <-- BEST" if r["auroc"] == best_auroc else ""
        print(
            f"{r['config']:<18s} {str(r['hidden_dims']):<24s} {r['lr']:>7.0e} "
            f"{r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['spearman']:>9.4f} "
            f"{r['best_val_loss']:>8.4f} {r['best_epoch']:>6d} "
            f"{r['total_time_s']/60:>7.1f}m{tag}"
        )

    print(f"\nPhase 2 reference (old 579 dims, LN_gelu): AUROC=0.6708, AUPR=0.3690")
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
