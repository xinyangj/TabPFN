#!/usr/bin/env python
"""Train interpretation model on DREAM4-aligned GNW data and evaluate on DREAM4-10.

Phases:
  1. Load aligned GNW training data from cache
  2. Compare feature distributions vs DREAM4-10 (KS distances)
  3. Train MLP interpretation model (slim 583d and full 1267d)
  4. Evaluate on pre-extracted DREAM4-10 features (5 networks)

Usage:
    python scripts/train_gnw_d4aligned.py --device cuda:2
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(".").resolve()))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:2"
CACHE_DIR = Path("data/interpretation_cache_gnw_d4aligned")
DREAM_DIR = Path("data/dream_interpretation_features/v6")
OUT_DIR = Path("results/interpretation_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MODE = "binary_direct"

CATEGORIES_FULL = sorted([
    "between_features_attention", "between_items_attention",
    "embeddings", "gradients", "mlp_activations",
])
DIM_FULL = 1267
MAX_FEATURES = 151

CATEGORIES_SLIM = sorted([
    "between_features_attention", "between_items_attention",
    "gradients", "mlp_activations",
])
# Slim: 327 (bf_attn) + 162 (bi_attn) + 4 (first 4 gradient dims) + 90 (mlp) = 583
DIM_SLIM = 583
# Gradient slice for slim mode: only first 4 of 112 gradient dimensions
GRAD_SLIM_SLICE = slice(0, 4)


# ── Data loading ──────────────────────────────────────────────────────

def load_gnw_data(cache_dir: Path, categories: list[str], dim: int, max_datasets: int = 0,
                  slim: bool = False):
    """Load GNW-generated datasets."""
    files = sorted(cache_dir.glob("dataset_*.npz"))
    if max_datasets > 0:
        files = files[:max_datasets]

    data = []
    skipped = 0
    for f in files:
        try:
            d = np.load(f)
            parts = []
            for c in categories:
                key = f"cat_{c}"
                if key not in d:
                    break
                cat_data = d[key]
                if slim and c == "gradients":
                    cat_data = cat_data[:, GRAD_SLIM_SLICE]
                parts.append(cat_data)
            if len(parts) != len(categories):
                skipped += 1
                continue
            fv = np.concatenate(parts, axis=1).astype(np.float32)
            if fv.shape[1] != dim:
                skipped += 1
                continue
            lab = d[f"label_{LABEL_MODE}"].astype(np.float32)
            data.append((fv, lab))
        except Exception:
            skipped += 1

    logger.info(f"Loaded {len(data)} datasets from {cache_dir} ({skipped} skipped)")
    return data


def split_data(data, train_frac=0.8, val_frac=0.1):
    """Split into train/val/test."""
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return data[:n_train], data[n_train:n_train + n_val], data[n_train + n_val:]


# ── Distribution comparison ──────────────────────────────────────────

def compute_ks_distances(gnw_data, dream_dir: Path, categories: list[str]):
    """Compute KS distances between GNW features and DREAM4-10 features per category."""
    # Collect GNW feature vectors
    gnw_all = np.concatenate([fv for fv, _ in gnw_data], axis=0)
    logger.info(f"GNW features: {gnw_all.shape}")

    # Collect DREAM4-10 feature vectors
    dream_all = []
    for net_id in range(1, 6):
        path = dream_dir / f"dream4_10_net{net_id}.npz"
        if not path.exists():
            continue
        d = np.load(path)
        for key in d.files:
            if key.startswith("fv_"):
                dream_all.append(d[key])
    if not dream_all:
        logger.warning("No DREAM4-10 features found!")
        return {}
    dream_all = np.concatenate(dream_all, axis=0)
    logger.info(f"DREAM4-10 features: {dream_all.shape}")

    # Compute per-category KS distances
    results = {}
    col_start = 0
    for cat in categories:
        # Get category dimension from first dataset
        d0 = np.load(sorted(Path(CACHE_DIR).glob("dataset_*.npz"))[0])
        cat_key = f"cat_{cat}"
        if cat_key not in d0:
            continue
        cat_dim = d0[cat_key].shape[1]

        gnw_cat = gnw_all[:, col_start:col_start + cat_dim].flatten()
        dream_cat = dream_all[:, col_start:col_start + cat_dim].flatten()
        col_start += cat_dim

        ks_stat, ks_p = ks_2samp(
            gnw_cat[np.isfinite(gnw_cat)][:100000],
            dream_cat[np.isfinite(dream_cat)][:100000],
        )
        results[cat] = {"ks_stat": float(ks_stat), "ks_p": float(ks_p), "dim": cat_dim}
        logger.info(f"  {cat} ({cat_dim}d): KS={ks_stat:.4f}")

    return results


# ── Dataset class ─────────────────────────────────────────────────────

class PaddedDataset(Dataset):
    def __init__(self, data_list, max_features, dim, augment=False):
        n = len(data_list)
        self.features = torch.zeros(n, max_features, dim)
        self.labels = torch.zeros(n, max_features)
        self.masks = torch.zeros(n, max_features)
        self.n_features = torch.zeros(n, dtype=torch.long)
        self.augment = augment

        for i, (fv, lab) in enumerate(data_list):
            nf = min(fv.shape[0], max_features)
            self.features[i, :nf] = torch.from_numpy(fv[:nf])
            self.labels[i, :nf] = torch.from_numpy(lab[:nf])
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
        return {
            "features": self.features[idx],
            "labels": self.labels[idx],
            "mask": self.masks[idx],
        }


# ── Training ──────────────────────────────────────────────────────────

def compute_norm_stats(dataset, n_sample=50000):
    """Compute global mean/std from dataset."""
    indices = np.random.choice(len(dataset), min(n_sample, len(dataset)), replace=False)
    all_f = []
    for idx in indices:
        item = dataset[int(idx)]
        nf = int(item["mask"].sum())
        if nf > 0:
            all_f.append(item["features"][:nf])
    cat = torch.cat(all_f, dim=0)
    return cat.mean(dim=0), cat.std(dim=0).clamp(min=1e-6)


def train_model(train_ds, val_ds, dim, config, device):
    """Train interpretation MLP."""
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss

    model = InterpretationModel(
        "mlp", input_dim=dim, output_mode="binary",
        hidden_dims=config["hidden_dims"], dropout=config["dropout"],
        norm=config["norm"], activation=config["activation"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

    loss_fn = InterpretationLoss(mode="binary").to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["max_epochs"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], num_workers=2, pin_memory=True,
    )

    gm, gs = compute_norm_stats(train_ds)
    gm_d, gs_d = gm.to(device), gs.to(device)

    best_vl, best_ep, patience_counter = float("inf"), 0, 0
    best_state = None

    for epoch in range(config["max_epochs"]):
        model.train()
        train_losses = []
        for batch in train_loader:
            f = (batch["features"].to(device) - gm_d) / gs_d
            logits = model(f).squeeze(-1)
            loss = loss_fn(logits, batch["labels"].to(device), batch["mask"].to(device).bool())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                f = (batch["features"].to(device) - gm_d) / gs_d
                logits = model(f).squeeze(-1)
                loss = loss_fn(
                    logits, batch["labels"].to(device), batch["mask"].to(device).bool(),
                )
                val_losses.append(loss.item())

        tl_avg = np.mean(train_losses)
        vl_avg = np.mean(val_losses)
        scheduler.step()

        if vl_avg < best_vl:
            best_vl = vl_avg
            best_ep = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            logger.info(
                f"  Epoch {epoch:3d}: train={tl_avg:.4f} val={vl_avg:.4f} "
                f"best={best_vl:.4f}@{best_ep} pat={patience_counter}"
            )

        if patience_counter >= config["patience"]:
            logger.info(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, gm, gs, best_vl, best_ep


# ── DREAM4-10 evaluation ─────────────────────────────────────────────

def evaluate_dream4_10(model, gm, gs, categories, dim, device):
    """Evaluate model on DREAM4-10 pre-extracted features."""
    gm_d, gs_d = gm.to(device), gs.to(device)

    all_aurocs, all_auprs = [], []
    net_results = []

    for net_id in range(1, 6):
        path = DREAM_DIR / f"dream4_10_net{net_id}.npz"
        if not path.exists():
            logger.warning(f"Missing {path}")
            continue

        data = np.load(path, allow_pickle=True)
        gene_names = list(data["gene_names"])
        gold_tf = list(data["gold_tf"])
        gold_target = list(data["gold_target"])
        gold_weight = data["gold_weight"]

        gold_edges = set()
        for i in range(len(gold_tf)):
            if gold_weight[i] > 0:
                gold_edges.add((gold_tf[i], gold_target[i]))

        target_genes = list(data["target_genes"])
        edge_scores = {}

        for target in target_genes:
            fv_key = f"fv_{target}"
            tfs_key = f"tfs_{target}"
            if fv_key not in data.files or tfs_key not in data.files:
                continue

            fv_full = data[fv_key]  # (n_tfs, 1267)
            tfs = list(data[tfs_key])

            # Select category columns
            if dim == DIM_FULL:
                fv = fv_full
            else:
                # Slim 583d: bf_attn(327) + bi_attn(162) + gradients[:4] + mlp(90)
                # In 1267d sorted order: bf_attn[0:327], bi_attn[327:489], embed[489:1065], grad[1065:1177], mlp[1177:1267]
                fv = np.concatenate([
                    fv_full[:, :327],         # between_features_attention (327)
                    fv_full[:, 327:489],      # between_items_attention (162)
                    fv_full[:, 1065:1069],    # gradients first 4 dims
                    fv_full[:, 1177:1267],    # mlp_activations (90)
                ], axis=1)

            if fv.shape[1] != dim:
                logger.warning(f"Dim mismatch: {fv.shape[1]} vs {dim}")
                continue

            fv_t = torch.tensor(fv, dtype=torch.float32).unsqueeze(0).to(device)
            fv_t = (fv_t - gm_d) / gs_d

            with torch.no_grad():
                logits = model(fv_t)
            scores = torch.sigmoid(logits).squeeze(0).squeeze(-1).cpu().numpy()

            for j, tf in enumerate(tfs):
                if tf != target:
                    edge_scores[(tf, target)] = float(scores[j])

        # Compute AUROC/AUPR
        all_possible = set()
        tf_names = list(data.get("tf_names", gene_names))
        for tf in tf_names:
            for tg in target_genes:
                if tf != tg:
                    all_possible.add((tf, tg))

        y_true, y_score = [], []
        for edge in sorted(all_possible):
            y_true.append(1 if edge in gold_edges else 0)
            y_score.append(edge_scores.get(edge, 0.0))

        y_true = np.array(y_true)
        y_score = np.array(y_score)

        if len(np.unique(y_true)) < 2:
            continue

        auroc = roc_auc_score(y_true, y_score)
        aupr = average_precision_score(y_true, y_score)
        all_aurocs.append(auroc)
        all_auprs.append(aupr)
        net_results.append({
            "network": f"net{net_id}",
            "auroc": float(auroc),
            "aupr": float(aupr),
            "n_edges": int(y_true.sum()),
            "n_total": len(y_true),
        })
        logger.info(f"  Net {net_id}: AUROC={auroc:.4f} AUPR={aupr:.4f}")

    avg_auroc = np.mean(all_aurocs) if all_aurocs else 0
    avg_aupr = np.mean(all_auprs) if all_auprs else 0
    logger.info(f"  DREAM4-10 Average: AUROC={avg_auroc:.4f} AUPR={avg_aupr:.4f}")

    return {
        "avg_auroc": float(avg_auroc),
        "avg_aupr": float(avg_aupr),
        "networks": net_results,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:2")
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--max_datasets", type=int, default=0)
    parser.add_argument("--skip_dist", action="store_true", help="Skip distribution comparison")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device
    cache_dir = Path(args.cache_dir)

    results = {}

    # ── Phase 1: Load data ──
    logger.info("=" * 60)
    logger.info("Phase 1: Loading DREAM4-aligned GNW data")
    logger.info("=" * 60)

    data_full = load_gnw_data(cache_dir, CATEGORIES_FULL, DIM_FULL, args.max_datasets)
    data_slim = load_gnw_data(cache_dir, CATEGORIES_SLIM, DIM_SLIM, args.max_datasets, slim=True)

    if len(data_full) < 100:
        logger.error(f"Not enough data: {len(data_full)} datasets")
        return

    # ── Phase 2: Distribution comparison ──
    if not args.skip_dist:
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Feature distribution comparison (KS distances)")
        logger.info("=" * 60)
        ks_results = compute_ks_distances(data_full, DREAM_DIR, CATEGORIES_FULL)
        results["ks_distances"] = ks_results

    # ── Phase 3: Train models ──
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Training interpretation models")
    logger.info("=" * 60)

    configs = {
        "slim_gnorm": {
            "categories": CATEGORIES_SLIM,
            "dim": DIM_SLIM,
            "hidden_dims": [1024, 512, 256, 128],
            "lr": 1e-4, "dropout": 0.1, "norm": "layer", "activation": "gelu",
            "batch_size": 1024, "max_epochs": 300, "patience": 30, "weight_decay": 1e-4,
        },
        "full_gnorm": {
            "categories": CATEGORIES_FULL,
            "dim": DIM_FULL,
            "hidden_dims": [1024, 512, 256, 128],
            "lr": 1e-4, "dropout": 0.1, "norm": "layer", "activation": "gelu",
            "batch_size": 1024, "max_epochs": 300, "patience": 30, "weight_decay": 1e-4,
        },
    }

    for cfg_name, cfg in configs.items():
        logger.info(f"\n--- Training: {cfg_name} ({cfg['dim']}d) ---")
        data = data_slim if cfg["dim"] == DIM_SLIM else data_full
        train_data, val_data, test_data = split_data(data)
        logger.info(f"Split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

        train_ds = PaddedDataset(train_data, MAX_FEATURES, cfg["dim"], augment=True)
        val_ds = PaddedDataset(val_data, MAX_FEATURES, cfg["dim"])
        test_ds = PaddedDataset(test_data, MAX_FEATURES, cfg["dim"])

        model, gm, gs, best_vl, best_ep = train_model(
            train_ds, val_ds, cfg["dim"], cfg, DEVICE,
        )

        # ── Phase 4: Evaluate on DREAM4-10 ──
        logger.info(f"\n--- Evaluating {cfg_name} on DREAM4-10 ---")
        dream_results = evaluate_dream4_10(
            model, gm, gs, cfg["categories"], cfg["dim"], DEVICE,
        )

        # Save checkpoint
        model_path = OUT_DIR / f"gnw_d4aligned_{cfg_name}.pt"
        torch.save({
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": cfg,
            "input_dim": cfg["dim"],
            "categories": cfg["categories"],
            "global_mean": gm,
            "global_std": gs,
            "best_val_loss": best_vl,
            "best_epoch": best_ep,
            "dream4_10": dream_results,
            "n_train": len(train_data),
            "data_source": "gnw_dream4_aligned",
            # For compatibility with load_interpretation_model
            "variant": "mlp",
            "output_mode": "binary",
            "hidden_dims": cfg["hidden_dims"],
            "dropout": cfg["dropout"],
            "norm": cfg["norm"],
            "activation": cfg["activation"],
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        }, model_path)
        logger.info(f"Saved: {model_path}")

        results[cfg_name] = {
            "best_val_loss": best_vl,
            "best_epoch": best_ep,
            "n_train": len(train_data),
            "dim": cfg["dim"],
            "dream4_10": dream_results,
        }

    # ── Summary ──
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Baseline (best_dream_slim_gnorm): DREAM4-10 AUROC = 0.629")
    for cfg_name, r in results.items():
        if cfg_name == "ks_distances":
            continue
        d4 = r.get("dream4_10", {})
        logger.info(
            f"{cfg_name} ({r['dim']}d, {r['n_train']} train): "
            f"DREAM4-10 AUROC = {d4.get('avg_auroc', 0):.4f}, "
            f"AUPR = {d4.get('avg_aupr', 0):.4f}"
        )

    # Save full results
    json_path = OUT_DIR / "gnw_d4aligned_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")


if __name__ == "__main__":
    main()
