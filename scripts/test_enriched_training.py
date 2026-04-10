#!/usr/bin/env python
"""Test enriched signal training with gradient ablation.

Runs multiple configs on the same 2% data to compare:
1. Current signals (no gradients): between_features_attn + between_items_attn + mlp_activations
2. Enriched signals + gradients: all above + gradients + items_attention_gradients
3. With/without input BatchNorm

Usage:
    python scripts/test_enriched_training.py --cache_dir /tmp/enriched_grad_2pct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
MAX_EPOCHS = 50
PATIENCE = 15
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
LABEL_MODE = "binary_direct"


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
            return {"features": features, "labels": labels, "mask": self.masks[idx], "n_features": self.n_features[idx]}
        return {"features": self.features[idx], "labels": self.labels[idx], "mask": self.masks[idx], "n_features": self.n_features[idx]}


def load_and_split(cache_dir: Path):
    from scripts.generate_interpretation_data import load_all_datasets
    all_data = load_all_datasets(cache_dir)
    n = len(all_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    return all_data[:n_train], all_data[n_train:n_train + n_val], all_data[n_train + n_val:]


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


def train_and_evaluate(train_sel, val_sel, test_sel, *, config_name, hidden_dims, lr, norm, activation, input_batch_norm, global_norm_stats=None):
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from tabpfn.interpretation.evaluation.metrics import evaluate_binary, evaluate_ranking

    max_features = max(r["feature_vectors"].shape[0] for r in train_sel + val_sel + test_sel) + 1
    input_dim = train_sel[0]["feature_vectors"].shape[1]

    train_ds = FastDataset(train_sel, LABEL_MODE, max_features, augment=True)
    val_ds = FastDataset(val_sel, LABEL_MODE, max_features, augment=False)
    test_ds = FastDataset(test_sel, LABEL_MODE, max_features, augment=False)

    model = InterpretationModel(
        "mlp", input_dim=input_dim, output_mode="binary",
        hidden_dims=hidden_dims, dropout=DROPOUT,
        norm=norm, activation=activation,
        input_batch_norm=input_batch_norm,
    ).to(DEVICE)

    if global_norm_stats is not None:
        mean_t = torch.tensor(global_norm_stats[0], dtype=torch.float32).to(DEVICE)
        std_t = torch.tensor(global_norm_stats[1], dtype=torch.float32).to(DEVICE)
        model.set_global_norm_stats(mean_t, std_t)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{config_name}] input_dim={input_dim}, params={n_params:,}, input_bn={input_batch_norm}")

    loss_fn = InterpretationLoss(mode="binary").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    best_epoch = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            features = batch["features"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
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
                features = batch["features"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)
                preds = model(features, mask=mask)
                loss = loss_fn(preds, labels, mask=mask.bool())
                val_loss_sum += loss.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            best_epoch = epoch
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    # Evaluate
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels, all_masks = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)
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

    return {
        "config": config_name,
        "input_dim": input_dim,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "train_loss": train_loss,
        "auroc": metrics_bin.get("auroc", 0),
        "aupr": metrics_bin.get("aupr", 0),
        "spearman": metrics_rank.get("spearman", 0) if metrics_rank else 0,
    }


def compute_global_norm_stats(data_records):
    """Compute per-dim mean and std from training data for z-score normalization."""
    all_vecs = []
    for record in data_records:
        fv = record["feature_vectors"]
        all_vecs.append(fv)
    stacked = np.concatenate(all_vecs, axis=0)  # (total_features, D)
    mean = stacked.mean(axis=0).astype(np.float32)
    std = stacked.std(axis=0).astype(np.float32)
    return mean, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    args = parser.parse_args()
    cache_dir = Path(args.cache_dir)

    print(f"Loading data from {cache_dir}...")
    train_data, val_data, test_data = load_and_split(cache_dir)
    print(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Select categories (no gradients — isolate normalization effect)
    BASE_CATS = {"between_features_attention", "between_items_attention", "mlp_activations"}
    train_sel = select_categories(train_data, BASE_CATS)
    val_sel = select_categories(val_data, BASE_CATS)
    test_sel = select_categories(test_data, BASE_CATS)
    print(f"After category selection: {len(train_sel)} train, {len(val_sel)} val, {len(test_sel)} test")
    print(f"Input dim: {train_sel[0]['feature_vectors'].shape[1]}")

    # Precompute global norm stats from training data
    print("Computing global normalization stats from training data...")
    global_mean, global_std = compute_global_norm_stats(train_sel)
    dead_dims = (global_std < 1e-6).sum()
    print(f"  Mean range: [{global_mean.min():.4f}, {global_mean.max():.4f}]")
    print(f"  Std range:  [{global_std.min():.6f}, {global_std.max():.4f}]")
    print(f"  Dead dims (std<1e-6): {dead_dims}")

    base_kwargs = dict(
        hidden_dims=[1024, 512, 256, 128],
        lr=1e-4,
        activation="gelu",
    )

    # 6-config ablation: {LN, BN} × {None, InputBN, GlobalNorm}
    experiments = [
        # (name, hidden_norm, input_batch_norm, use_global_norm)
        ("LN",              "layer", False, False),
        ("LN+InputBN",      "layer", True,  False),
        ("LN+GlobalNorm",   "layer", False, True),
        ("BN",              "batch", False, False),
        ("BN+InputBN",      "batch", True,  False),
        ("BN+GlobalNorm",   "batch", False, True),
    ]

    results = []
    for exp_name, hidden_norm, use_input_bn, use_global in experiments:
        gn_stats = (global_mean, global_std) if use_global else None
        r = train_and_evaluate(
            train_sel, val_sel, test_sel,
            config_name=exp_name,
            norm=hidden_norm,
            input_batch_norm=use_input_bn,
            global_norm_stats=gn_stats,
            **base_kwargs,
        )
        results.append(r)

    # Print results table
    print(f"\n{'='*85}")
    print(f"NORMALIZATION ABLATION (2% data, no_grad, binary_direct)")
    print(f"{'='*85}")
    print(f"{'Config':<20s} {'Dims':>6s} {'Params':>10s} {'AUROC':>8s} {'AUPR':>8s} {'Spearman':>9s} {'ValLoss':>8s} {'Epoch':>6s}")
    print("-" * 85)
    for r in results:
        print(f"{r['config']:<20s} {r['input_dim']:>6d} {r['n_params']:>10,d} "
              f"{r['auroc']:>8.4f} {r['aupr']:>8.4f} {r['spearman']:>9.4f} "
              f"{r['best_val_loss']:>8.4f} {r['best_epoch']:>6d}")

    print(f"\nReference: Phase 2 best (B2, LN, 579 dims, 100% data): AUROC=0.6708, AUPR=0.3690")

    # Save results
    out_path = Path("results/interpretation_experiments/norm_ablation_2pct.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

