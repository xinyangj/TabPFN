#!/usr/bin/env python
"""Train interpretation model on 300K+ v6 datasets."""
import json, logging, time, sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path('.').resolve()))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda:1"
CACHE_DIR = Path("data/interpretation_cache_v6_1m")
OUT_DIR = Path("results/interpretation_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_MODE = "binary_direct"
DIM = 1267
MAX_FEATURES = 151

CONFIG = {
    "hidden_dims": [1024, 512, 256, 128],
    "lr": 1e-4,
    "dropout": 0.1,
    "norm": "layer",
    "activation": "gelu",
    "batch_size": 1024,
    "max_epochs": 100,
    "patience": 15,
    "weight_decay": 1e-4,
}

V6_CATEGORIES = sorted([
    "between_features_attention", "between_items_attention",
    "embeddings", "mlp_activations", "gradients",
])


class PreloadedDataset(Dataset):
    def __init__(self, features, labels, masks, n_features, augment=False):
        self.features = features
        self.labels = labels
        self.masks = masks
        self.n_features = n_features
        self.augment = augment

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


def load_all_data(cache_dir, categories, label_mode, dim):
    files = sorted(cache_dir.glob("dataset_*.npz"))
    logger.info(f"Loading {len(files)} files from {cache_dir}...")
    data = []
    skipped = 0
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            d = np.load(f)
            parts = [d[f"cat_{c}"] for c in categories if f"cat_{c}" in d]
            if not parts:
                skipped += 1; continue
            fv = np.concatenate(parts, axis=1).astype(np.float32)
            if fv.shape[1] != dim:
                skipped += 1; continue
            lab = d[f"label_{label_mode}"].astype(np.float32)
            if lab.sum() == 0:
                skipped += 1; continue
            data.append((fv, lab))
        except Exception:
            skipped += 1
        if (i + 1) % 50000 == 0:
            logger.info(f"  {i+1}/{len(files)} ({len(data)} valid, {skipped} skip) {time.time()-t0:.0f}s")
    logger.info(f"Loaded {len(data)} datasets ({skipped} skipped) in {time.time()-t0:.0f}s")
    return data


def build_padded_tensors(data_list, max_features, dim):
    n = len(data_list)
    features = torch.zeros(n, max_features, dim)
    labels = torch.zeros(n, max_features)
    masks = torch.zeros(n, max_features)
    nf_arr = torch.zeros(n, dtype=torch.long)
    for i, (fv, lab) in enumerate(data_list):
        nf = min(fv.shape[0], max_features)
        features[i, :nf] = torch.from_numpy(fv[:nf])
        labels[i, :nf] = torch.from_numpy(lab[:nf])
        masks[i, :nf] = 1.0
        nf_arr[i] = nf
        if (i + 1) % 100000 == 0:
            logger.info(f"  Padded {i+1}/{n}")
    return features, labels, masks, nf_arr


def compute_norm_stats(dataset, n_sample=50000):
    indices = np.random.choice(len(dataset), min(n_sample, len(dataset)), replace=False)
    all_f = []
    for idx in indices:
        item = dataset[int(idx)]
        nf = int(item["mask"].sum())
        if nf > 0:
            all_f.append(item["features"][:nf])
    cat = torch.cat(all_f, dim=0)
    return cat.mean(dim=0), cat.std(dim=0).clamp(min=1e-6)


def main():
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from scipy.stats import spearmanr

    cfg = CONFIG

    all_data = load_all_data(CACHE_DIR, V6_CATEGORIES, LABEL_MODE, DIM)

    rng = np.random.default_rng(42)
    indices = rng.permutation(len(all_data))
    n = len(all_data)
    n_train, n_val = int(n * 0.8), int(n * 0.1)

    train_data = [all_data[i] for i in indices[:n_train]]
    val_data = [all_data[i] for i in indices[n_train:n_train + n_val]]
    test_data = [all_data[i] for i in indices[n_train + n_val:]]
    del all_data

    max_feat = max(
        max(d[0].shape[0] for d in train_data),
        max(d[0].shape[0] for d in val_data),
        max(d[0].shape[0] for d in test_data),
    )
    max_features = min(max_feat + 1, MAX_FEATURES)
    logger.info(f"Split: {len(train_data)}/{len(val_data)}/{len(test_data)}, max_feat={max_features}")

    logger.info("Building padded tensors...")
    t0 = time.time()
    tr_f, tr_l, tr_m, tr_nf = build_padded_tensors(train_data, max_features, DIM); del train_data
    va_f, va_l, va_m, va_nf = build_padded_tensors(val_data, max_features, DIM); del val_data
    te_f, te_l, te_m, te_nf = build_padded_tensors(test_data, max_features, DIM); del test_data
    logger.info(f"Padded in {time.time()-t0:.0f}s, train={tr_f.shape} ({tr_f.nelement()*4/1e9:.1f}GB)")

    train_ds = PreloadedDataset(tr_f, tr_l, tr_m, tr_nf, augment=True)
    val_ds = PreloadedDataset(va_f, va_l, va_m, va_nf)
    test_ds = PreloadedDataset(te_f, te_l, te_m, te_nf)

    logger.info("Computing norm stats...")
    gm, gs = compute_norm_stats(train_ds)

    model = InterpretationModel(
        "mlp", input_dim=DIM, output_mode="binary",
        hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
        norm=cfg["norm"], activation=cfg["activation"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['hidden_dims']}, params={n_params:,}")

    loss_fn = InterpretationLoss(mode="binary").to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], num_workers=2, pin_memory=True)

    gm_d, gs_d = gm.to(DEVICE), gs.to(DEVICE)
    best_vl, best_ep, pat = float("inf"), 0, 0
    t_start = time.time()

    for epoch in range(cfg["max_epochs"]):
        model.train()
        tl = []
        for batch in train_loader:
            f = (batch["features"].to(DEVICE) - gm_d) / gs_d
            logits = model(f).squeeze(-1)
            loss = loss_fn(logits, batch["labels"].to(DEVICE), batch["mask"].to(DEVICE).bool())
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for batch in val_loader:
                f = (batch["features"].to(DEVICE) - gm_d) / gs_d
                logits = model(f).squeeze(-1)
                loss = loss_fn(logits, batch["labels"].to(DEVICE), batch["mask"].to(DEVICE).bool())
                vl.append(loss.item())

        val_loss = np.mean(vl); scheduler.step()
        improved = val_loss < best_vl
        if improved:
            best_vl, best_ep, pat = val_loss, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "global_mean": gm, "global_std": gs,
                "config": cfg, "input_dim": DIM, "max_features": max_features,
                "label_mode": LABEL_MODE, "categories": V6_CATEGORIES,
            }, OUT_DIR / "best_v6_300k_model.pt")
        else:
            pat += 1

        ep_time = time.time() - t_start
        if epoch % 5 == 0 or improved:
            logger.info(f"ep {epoch:3d}: train={np.mean(tl):.4f} val={val_loss:.4f} best={best_vl:.4f}@{best_ep} "
                        f"lr={optimizer.param_groups[0]['lr']:.1e} ({ep_time/60:.1f}min){' *' if improved else ''}")

        if pat >= cfg["patience"]:
            logger.info(f"Early stopping at epoch {epoch}"); break

    total_time = time.time() - t_start
    logger.info(f"Training: {total_time/60:.1f}min, best epoch={best_ep}")

    # Evaluate
    ckpt = torch.load(OUT_DIR / "best_v6_300k_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    preds, labs = [], []
    with torch.no_grad():
        for batch in test_loader:
            f = (batch["features"].to(DEVICE) - gm_d) / gs_d
            p = torch.sigmoid(model(f).squeeze(-1))
            m = batch["mask"]
            for i in range(f.shape[0]):
                nf = int(m[i].sum())
                if nf > 0:
                    preds.append(p[i, :nf].cpu().numpy())
                    labs.append(batch["labels"][i, :nf].numpy())

    pf, lf = np.concatenate(preds), np.concatenate(labs)
    auroc = roc_auc_score(lf, pf)
    aupr = average_precision_score(lf, pf)
    f1 = f1_score(lf, (pf > 0.5).astype(int), zero_division=0)
    sps = [spearmanr(p, l)[0] for p, l in zip(preds, labs)
           if len(np.unique(l)) > 1 and len(l) > 1 and not np.isnan(spearmanr(p, l)[0])]
    sp_avg = np.mean(sps) if sps else 0

    logger.info(f"\n=== Test Results (300K v6) ===")
    logger.info(f"AUROC:    {auroc:.4f}")
    logger.info(f"AUPR:     {aupr:.4f}")
    logger.info(f"F1:       {f1:.4f}")
    logger.info(f"Spearman: {sp_avg:.4f}")
    logger.info(f"N_test: {len(preds)} datasets, {len(pf)} features, pos_rate={lf.mean():.3f}")

    summary = {
        "version": "v6_300k", "input_dim": DIM, "categories": V6_CATEGORIES,
        "n_total": len(train_ds) + len(val_ds) + len(test_ds),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "config": cfg, "best_epoch": best_ep, "best_val_loss": float(best_vl),
        "total_time_s": total_time, "n_params": n_params,
        "test_metrics": {"auroc": float(auroc), "aupr": float(aupr), "f1": float(f1), "spearman": float(sp_avg)},
    }
    with open(OUT_DIR / "best_v6_300k_model.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved to {OUT_DIR}/best_v6_300k_model.*")

if __name__ == "__main__":
    main()
