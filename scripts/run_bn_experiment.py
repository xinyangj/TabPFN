#!/usr/bin/env python
"""Run B2 config with BatchNorm (instead of LayerNorm) on the full 141K dataset.

Replicates Phase 2 best config exactly, only changing norm="layer" → norm="batch".

Usage:
    python scripts/run_bn_experiment.py
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
CACHE_DIR = Path("data/interpretation_cache")

# Exact Phase 2 hyperparams
BATCH_SIZE = 512
MAX_EPOCHS = 200
PATIENCE = 25
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
LABEL_MODE = "binary_direct"
INPUT_CATEGORIES = {"between_features_attention", "between_items_attention", "mlp_activations"}


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


def select_categories(data, categories):
    result = []
    for record in data:
        cat_vecs = record.get("category_vectors", {})
        parts = [cat_vecs[c] for c in sorted(categories) if c in cat_vecs]
        if not parts:
            continue
        fv = np.concatenate(parts, axis=1)
        result.append({"feature_vectors": fv, "labels": record["labels"], "metadata": record["metadata"]})
    return result


def train_config(name, norm, train_ds, val_ds, test_ds, input_dim):
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.evaluation.metrics import evaluate_binary, evaluate_ranking

    model = InterpretationModel(
        "mlp", input_dim=input_dim, output_mode="binary",
        hidden_dims=[1024, 512, 256, 128], dropout=DROPOUT,
        norm=norm, activation="gelu",
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{name}] norm={norm}, input_dim={input_dim}, params={n_params:,}")

    loss_fn = InterpretationLoss(mode="binary").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

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
            logger.info(f"  [{name}] epoch {epoch:3d} train={train_loss:.4f} val={val_loss:.4f} best={best_val_loss:.4f} ({elapsed:.0f}s){marker}")

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
    return {
        "config": name,
        "norm": norm,
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
    logger.info("Loading full 141K dataset...")
    train_data, val_data, test_data = load_and_split_data()

    train_sel = select_categories(train_data, INPUT_CATEGORIES)
    val_sel = select_categories(val_data, INPUT_CATEGORIES)
    test_sel = select_categories(test_data, INPUT_CATEGORIES)
    logger.info(f"After selection: {len(train_sel)} train, {len(val_sel)} val, {len(test_sel)} test")

    input_dim = train_sel[0]["feature_vectors"].shape[1]
    max_features = max(r["feature_vectors"].shape[0] for r in train_sel + val_sel + test_sel) + 1
    logger.info(f"Input dim: {input_dim}, max_features: {max_features}")

    train_ds = FastDataset(train_sel, LABEL_MODE, max_features, augment=True)
    val_ds = FastDataset(val_sel, LABEL_MODE, max_features, augment=False)
    test_ds = FastDataset(test_sel, LABEL_MODE, max_features, augment=False)
    logger.info(f"Datasets tensorized: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Run both LN (reference) and BN (experiment)
    results = []
    for name, norm in [("B2_BN_gelu", "batch"), ("B2_LN_gelu_ref", "layer")]:
        r = train_config(name, norm, train_ds, val_ds, test_ds, input_dim)
        results.append(r)
        logger.info(f"[{name}] AUROC={r['auroc']:.4f} AUPR={r['aupr']:.4f} Spearman={r['spearman']:.4f}")

    # Print comparison
    print(f"\n{'='*90}")
    print(f"BN vs LN COMPARISON (full 141K dataset, binary_direct, 579 dims)")
    print(f"{'='*90}")
    print(f"{'Config':<20s} {'Norm':>6s} {'AUROC':>8s} {'AUPR':>8s} {'F1':>8s} {'Spearman':>9s} {'ValLoss':>8s} {'Epoch':>6s} {'Time':>8s}")
    print("-" * 90)
    for r in results:
        print(f"{r['config']:<20s} {r['norm']:>6s} {r['auroc']:>8.4f} {r['aupr']:>8.4f} "
              f"{r['f1']:>8.4f} {r['spearman']:>9.4f} {r['best_val_loss']:>8.4f} "
              f"{r['best_epoch']:>6d} {r['total_time_s']/60:>7.1f}m")

    print(f"\nPhase 2 reference (B2_LN_gelu): AUROC=0.6708, AUPR=0.3690")

    # Save
    out_path = Path("results/interpretation_experiments/bn_vs_ln_full.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
