#!/usr/bin/env python
"""Train interpretation models: Config A (current) and Config B (deeper) with patience=30, max_epochs=150."""
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
SPLIT_FILE = Path("data/interpretation_splits/v6_fixed_split.json")
OUT_DIR = Path("results/interpretation_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_MODE = "binary_direct"
DIM = 1267
MAX_FEATURES = 151

CONFIGS = {
    "A": {
        "name": "current_p30",
        "hidden_dims": [1024, 512, 256, 128],
        "lr": 1e-4, "dropout": 0.1, "norm": "layer", "activation": "gelu",
        "batch_size": 1024, "max_epochs": 150, "patience": 30, "weight_decay": 1e-4,
        "out_model": "best_v6_600k_p30_model.pt",
        "out_json": "best_v6_600k_p30_model.json",
    },
    "B": {
        "name": "deep_p30",
        "hidden_dims": [2048, 1024, 512, 256, 128],
        "lr": 1e-4, "dropout": 0.1, "norm": "layer", "activation": "gelu",
        "batch_size": 1024, "max_epochs": 150, "patience": 30, "weight_decay": 1e-4,
        "out_model": "best_v6_600k_deep_model.pt",
        "out_json": "best_v6_600k_deep_model.json",
    },
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


def load_split_data(cache_dir, split_ids, categories, label_mode, dim):
    data = []
    skipped = 0
    for seed_id in split_ids:
        f = cache_dir / f"dataset_{seed_id:06d}.npz"
        if not f.exists():
            skipped += 1; continue
        try:
            d = np.load(f)
            parts = [d[f"cat_{c}"] for c in categories if f"cat_{c}" in d]
            if not parts:
                skipped += 1; continue
            fv = np.concatenate(parts, axis=1).astype(np.float32)
            if fv.shape[1] != dim:
                skipped += 1; continue
            lab = d[f"label_{label_mode}"].astype(np.float32)
            data.append((fv, lab))
        except Exception:
            skipped += 1
    return data, skipped


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


def evaluate_model(model, loader, loss_fn, gm_d, gs_d, device):
    model.eval()
    losses, preds, labs = [], [], []
    with torch.no_grad():
        for batch in loader:
            f = (batch["features"].to(device) - gm_d) / gs_d
            logits = model(f).squeeze(-1)
            loss = loss_fn(logits, batch["labels"].to(device), batch["mask"].to(device).bool())
            losses.append(loss.item())
            p = torch.sigmoid(logits)
            m = batch["mask"]
            for i in range(f.shape[0]):
                nf = int(m[i].sum())
                if nf > 0:
                    preds.append(p[i, :nf].cpu().numpy())
                    labs.append(batch["labels"][i, :nf].numpy())
    return np.mean(losses), preds, labs


def compute_metrics(preds, labs):
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    from scipy.stats import spearmanr
    pf, lf = np.concatenate(preds), np.concatenate(labs)
    auroc = roc_auc_score(lf, pf)
    aupr = average_precision_score(lf, pf)
    f1 = f1_score(lf, (pf > 0.5).astype(int), zero_division=0)
    sps = [spearmanr(p, l)[0] for p, l in zip(preds, labs)
           if len(np.unique(l)) > 1 and len(l) > 1 and not np.isnan(spearmanr(p, l)[0])]
    sp_avg = np.mean(sps) if sps else 0
    return {"auroc": float(auroc), "aupr": float(aupr), "f1": float(f1),
            "spearman": float(sp_avg), "n_datasets": len(preds),
            "n_features": len(pf), "pos_rate": float(lf.mean())}


def train_config(cfg, train_ds, val_ds, test_ds, gm, gs):
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss

    logger.info(f"\n{'='*60}")
    logger.info(f"Training Config: {cfg['name']} — {cfg['hidden_dims']}")
    logger.info(f"patience={cfg['patience']}, max_epochs={cfg['max_epochs']}")
    logger.info(f"{'='*60}")

    model = InterpretationModel(
        "mlp", input_dim=DIM, output_mode="binary",
        hidden_dims=cfg["hidden_dims"], dropout=cfg["dropout"],
        norm=cfg["norm"], activation=cfg["activation"],
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params: {n_params:,}")

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
                "config": cfg, "input_dim": DIM, "max_features": MAX_FEATURES,
                "label_mode": LABEL_MODE, "categories": V6_CATEGORIES,
                "split_file": str(SPLIT_FILE),
            }, OUT_DIR / cfg["out_model"])
        else:
            pat += 1

        ep_time = time.time() - t_start
        if epoch % 5 == 0 or improved:
            logger.info(f"[{cfg['name']}] ep {epoch:3d}: train={np.mean(tl):.4f} val={val_loss:.4f} "
                        f"best={best_vl:.4f}@{best_ep} lr={optimizer.param_groups[0]['lr']:.1e} "
                        f"({ep_time/60:.1f}min){' *' if improved else ''}")

        if pat >= cfg["patience"]:
            logger.info(f"[{cfg['name']}] Early stopping at epoch {epoch}"); break

    total_time = time.time() - t_start
    logger.info(f"[{cfg['name']}] Training done: {total_time/60:.1f}min, best epoch={best_ep}")

    # Evaluate
    ckpt = torch.load(OUT_DIR / cfg["out_model"], weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    _, preds, labs = evaluate_model(model, test_loader, loss_fn, gm_d, gs_d, DEVICE)
    metrics = compute_metrics(preds, labs)

    logger.info(f"\n=== Test Results: {cfg['name']} ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    summary = {
        "version": f"v6_600k_{cfg['name']}", "input_dim": DIM, "categories": V6_CATEGORIES,
        "split_file": str(SPLIT_FILE),
        "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
        "config": cfg, "best_epoch": best_ep, "best_val_loss": float(best_vl),
        "total_time_s": total_time, "n_params": n_params,
        "test_metrics": metrics,
    }
    with open(OUT_DIR / cfg["out_json"], "w") as f:
        json.dump(summary, f, indent=2)

    # Free GPU memory
    del model, loss_fn, optimizer, scheduler
    torch.cuda.empty_cache()

    return metrics


def main():
    # Load fixed split
    with open(SPLIT_FILE) as f:
        split = json.load(f)
    logger.info(f"Fixed split: train={split['n_train']}, val={split['n_val']}, test={split['n_test']}")

    t0 = time.time()
    logger.info("Loading train data...")
    train_data, tr_skip = load_split_data(CACHE_DIR, split["train_ids"], V6_CATEGORIES, LABEL_MODE, DIM)
    logger.info(f"  train: {len(train_data)} loaded ({tr_skip} skip) in {time.time()-t0:.0f}s")

    logger.info("Loading val data...")
    val_data, va_skip = load_split_data(CACHE_DIR, split["val_ids"], V6_CATEGORIES, LABEL_MODE, DIM)
    logger.info(f"  val: {len(val_data)} loaded ({va_skip} skip) in {time.time()-t0:.0f}s")

    logger.info("Loading test data...")
    test_data, te_skip = load_split_data(CACHE_DIR, split["test_ids"], V6_CATEGORIES, LABEL_MODE, DIM)
    logger.info(f"  test: {len(test_data)} loaded ({te_skip} skip) in {time.time()-t0:.0f}s")

    logger.info("Building padded tensors...")
    t1 = time.time()
    tr_f, tr_l, tr_m, tr_nf = build_padded_tensors(train_data, MAX_FEATURES, DIM); del train_data
    va_f, va_l, va_m, va_nf = build_padded_tensors(val_data, MAX_FEATURES, DIM); del val_data
    te_f, te_l, te_m, te_nf = build_padded_tensors(test_data, MAX_FEATURES, DIM); del test_data
    logger.info(f"Padded in {time.time()-t1:.0f}s, train={tr_f.shape}")

    train_ds = PreloadedDataset(tr_f, tr_l, tr_m, tr_nf, augment=True)
    val_ds = PreloadedDataset(va_f, va_l, va_m, va_nf)
    test_ds = PreloadedDataset(te_f, te_l, te_m, te_nf)

    logger.info("Computing norm stats from train set...")
    gm, gs = compute_norm_stats(train_ds)

    # Train both configs
    results = {}
    for key in ["A", "B"]:
        cfg = CONFIGS[key]
        results[key] = train_config(cfg, train_ds, val_ds, test_ds, gm, gs)

    # Final comparison
    logger.info(f"\n{'='*60}")
    logger.info(f"=== FINAL COMPARISON (fixed test set, 52.7K datasets) ===")
    logger.info(f"{'='*60}")
    logger.info(f"{'Config':<20} {'AUROC':>8} {'AUPR':>8} {'F1':>8} {'Spearman':>10}")
    logger.info("-" * 60)
    
    # Include previous p15 result
    prev = json.load(open(OUT_DIR / "best_v6_600k_model.json"))
    logger.info(f"{'600K p15 [1024..]':<20} {prev['test_metrics_600k']['auroc']:>8.4f} "
                f"{prev['test_metrics_600k']['aupr']:>8.4f} {prev['test_metrics_600k']['f1']:>8.4f} "
                f"{prev['test_metrics_600k']['spearman']:>10.4f}")
    
    for key in ["A", "B"]:
        name = CONFIGS[key]["name"]
        m = results[key]
        logger.info(f"{name:<20} {m['auroc']:>8.4f} {m['aupr']:>8.4f} {m['f1']:>8.4f} {m['spearman']:>10.4f}")

    logger.info(f"\nAll results saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()
