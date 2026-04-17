#!/usr/bin/env python
"""Train GNW-D4aligned interpretation models and evaluate on DREAM4-10.

4 model variants × 5 data scales = 20 training runs.
Variants: slim (583d), full (1267d), grad_only (112d), slim_nonorm (583d).
Scales: 10K, 20K, 30K, 40K, 50K datasets.

Usage:
    python scripts/train_dream4_gnw.py --device cuda:0
    python scripts/train_dream4_gnw.py --device cuda:0 --variants slim grad_only --scales 10000 50000
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(".").resolve()))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/interpretation_cache_gnw_d4aligned")
DREAM_DIR = Path("data/dream_interpretation_features/v6")
OUT_DIR = Path("results/interpretation_experiments")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MODE = "binary_direct"
MAX_FEATURES = 151
SPLIT_SEED = 42

# Category dimensions in sorted alpha order (matching 1267d layout)
CAT_DIMS = {
    "between_features_attention": 327,
    "between_items_attention": 162,
    "embeddings": 576,
    "gradients": 112,
    "mlp_activations": 90,
}
ALL_CATS_SORTED = sorted(CAT_DIMS.keys())

# Compute offsets from canonical schema
CAT_OFFSETS = {}
_off = 0
for _c in ALL_CATS_SORTED:
    CAT_OFFSETS[_c] = (_off, _off + CAT_DIMS[_c])
    _off += CAT_DIMS[_c]
DIM_FULL = _off  # 1267


def make_feature_spec(variant: str) -> dict:
    """Return feature spec: categories, dim, grad_slice, and 1267d->variant slice ranges."""
    if variant in ("slim", "slim_nonorm"):
        cats = ["between_features_attention", "between_items_attention", "gradients", "mlp_activations"]
        grad_slice = slice(0, 4)  # only first 4 of 112 gradient dims
        dim = 327 + 162 + 4 + 90  # 583
        # Slices into 1267d full features for DREAM4-10 eval
        dream_slices = [
            CAT_OFFSETS["between_features_attention"],   # (0, 327)
            CAT_OFFSETS["between_items_attention"],       # (327, 489)
            (CAT_OFFSETS["gradients"][0], CAT_OFFSETS["gradients"][0] + 4),  # first 4 grad dims
            CAT_OFFSETS["mlp_activations"],               # (1177, 1267)
        ]
    elif variant == "full":
        cats = ALL_CATS_SORTED
        grad_slice = None
        dim = DIM_FULL
        dream_slices = [CAT_OFFSETS[c] for c in ALL_CATS_SORTED]
    elif variant == "grad_only":
        cats = ["gradients"]
        grad_slice = None  # all 112 dims
        dim = 112
        dream_slices = [CAT_OFFSETS["gradients"]]  # (1065, 1177)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    return {
        "categories": cats,
        "dim": dim,
        "grad_slice": grad_slice,
        "dream_slices": dream_slices,
        "use_global_norm": variant != "slim_nonorm",
    }


# ── Data loading ──────────────────────────────────────────────────────

def load_gnw_data(cache_dir: Path, spec: dict, file_list: list[Path]) -> list:
    """Load GNW datasets from a specific file list."""
    data = []
    skipped = 0
    for f in file_list:
        try:
            d = np.load(f)
            parts = []
            for c in spec["categories"]:
                key = f"cat_{c}"
                if key not in d:
                    break
                cat_data = d[key]
                if c == "gradients" and spec["grad_slice"] is not None:
                    cat_data = cat_data[:, spec["grad_slice"]]
                parts.append(cat_data)
            if len(parts) != len(spec["categories"]):
                skipped += 1
                continue
            fv = np.concatenate(parts, axis=1).astype(np.float32)
            if fv.shape[1] != spec["dim"]:
                skipped += 1
                continue
            lab = d[f"label_{LABEL_MODE}"].astype(np.float32)
            data.append((fv, lab))
        except Exception:
            skipped += 1
    if skipped > 0:
        logger.info(f"  Loaded {len(data)}, skipped {skipped}")
    return data


def create_splits(cache_dir: Path, seed: int = SPLIT_SEED):
    """Create shuffled file split: 80% train pool, 10% val, 10% test. Fixed seed."""
    files = sorted(cache_dir.glob("dataset_*.npz"))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(files))

    n_val = int(len(files) * 0.1)
    n_test = int(len(files) * 0.1)

    val_idx = indices[:n_val]
    test_idx = indices[n_val:n_val + n_test]
    train_pool_idx = indices[n_val + n_test:]

    files_arr = np.array(files)
    return {
        "train_pool": list(files_arr[train_pool_idx]),
        "val": list(files_arr[val_idx]),
        "test": list(files_arr[test_idx]),
        "n_total": len(files),
    }


# ── Dataset class ─────────────────────────────────────────────────────

class PaddedDataset(Dataset):
    def __init__(self, data_list, max_features, dim, augment=False):
        n = len(data_list)
        self.features = torch.zeros(n, max_features, dim)
        self.labels = torch.zeros(n, max_features)
        self.masks = torch.zeros(n, max_features)
        self.augment = augment

        for i, (fv, lab) in enumerate(data_list):
            nf = min(fv.shape[0], max_features)
            self.features[i, :nf] = torch.from_numpy(fv[:nf])
            self.labels[i, :nf] = torch.from_numpy(lab[:nf])
            self.masks[i, :nf] = 1.0

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        if self.augment:
            nf = int(self.masks[idx].sum())
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


def train_model(train_ds, val_ds, dim, config, device, use_global_norm=True):
    """Train interpretation MLP. Returns model, norm stats, best val loss, best epoch."""
    from tabpfn.interpretation.model.interpretation_model import InterpretationModel
    from tabpfn.interpretation.model.losses import InterpretationLoss

    model = InterpretationModel(
        "mlp", input_dim=dim, output_mode="binary",
        hidden_dims=config["hidden_dims"], dropout=config["dropout"],
        norm=config["norm"], activation=config["activation"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Model params: {n_params:,}")

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

    if use_global_norm:
        gm, gs = compute_norm_stats(train_ds)
        gm_d, gs_d = gm.to(device), gs.to(device)
    else:
        gm = torch.zeros(dim)
        gs = torch.ones(dim)
        gm_d = gm.to(device)
        gs_d = gs.to(device)

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
                f"    Ep {epoch:3d}: train={tl_avg:.4f} val={vl_avg:.4f} "
                f"best={best_vl:.4f}@{best_ep} pat={patience_counter}"
            )

        if patience_counter >= config["patience"]:
            logger.info(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    model.eval()
    return model, gm, gs, best_vl, best_ep


# ── DREAM4-10 evaluation ─────────────────────────────────────────────

def evaluate_dream4_10(model, gm, gs, spec, device):
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

            # Slice features according to variant's dream_slices
            parts = [fv_full[:, s:e] for s, e in spec["dream_slices"]]
            fv = np.concatenate(parts, axis=1)

            if fv.shape[1] != spec["dim"]:
                logger.warning(f"Dim mismatch: {fv.shape[1]} vs {spec['dim']}")
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

    avg_auroc = np.mean(all_aurocs) if all_aurocs else 0
    avg_aupr = np.mean(all_auprs) if all_auprs else 0

    return {
        "avg_auroc": float(avg_auroc),
        "avg_aupr": float(avg_aupr),
        "std_auroc": float(np.std(all_aurocs)) if all_aurocs else 0,
        "std_aupr": float(np.std(all_auprs)) if all_auprs else 0,
        "networks": net_results,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cache_dir", type=str, default=str(CACHE_DIR))
    parser.add_argument("--variants", nargs="+",
                        default=["slim", "full", "grad_only", "slim_nonorm"],
                        choices=["slim", "full", "grad_only", "slim_nonorm"])
    parser.add_argument("--scales", nargs="+", type=int,
                        default=[10000, 20000, 30000, 40000, 50000])
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    device = args.device
    cache_dir = Path(args.cache_dir)

    train_config = {
        "hidden_dims": [1024, 512, 256, 128],
        "lr": 1e-4, "dropout": 0.1, "norm": "layer", "activation": "gelu",
        "batch_size": 1024, "max_epochs": 300, "patience": 30, "weight_decay": 1e-4,
    }

    # ── Create fixed splits ──
    logger.info("Creating fixed train/val/test splits (seed=%d)", args.seed)
    splits = create_splits(cache_dir, seed=args.seed)
    logger.info(
        f"Total files: {splits['n_total']}, "
        f"train_pool: {len(splits['train_pool'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    all_results = {"seed": args.seed, "train_config": train_config, "runs": {}}

    for variant in args.variants:
        spec = make_feature_spec(variant)
        logger.info(f"\n{'='*60}")
        logger.info(f"Variant: {variant} ({spec['dim']}d, norm={spec['use_global_norm']})")
        logger.info(f"Categories: {spec['categories']}")
        logger.info(f"{'='*60}")

        # Load val/test once per variant
        val_data = load_gnw_data(cache_dir, spec, splits["val"])
        test_data = load_gnw_data(cache_dir, spec, splits["test"])
        val_ds = PaddedDataset(val_data, MAX_FEATURES, spec["dim"])

        logger.info(f"Val: {len(val_data)}, Test: {len(test_data)}")

        best_scale_auroc = 0
        best_scale_model = None

        for scale in sorted(args.scales):
            # Nested train subsets: first `scale` files from the shuffled train pool
            # This ensures 10K ⊂ 20K ⊂ 30K ⊂ 40K ⊂ 50K
            train_files = splits["train_pool"][:scale]
            actual_scale = len(train_files)
            if actual_scale < scale * 0.9:
                logger.warning(f"Requested {scale} but only {actual_scale} train files available")

            logger.info(f"\n--- {variant} @ {actual_scale} datasets ---")
            train_data = load_gnw_data(cache_dir, spec, train_files)
            logger.info(f"Loaded {len(train_data)} train datasets")

            if len(train_data) < 100:
                logger.warning(f"Too few datasets ({len(train_data)}), skipping")
                continue

            train_ds = PaddedDataset(train_data, MAX_FEATURES, spec["dim"], augment=True)

            model, gm, gs, best_vl, best_ep = train_model(
                train_ds, val_ds, spec["dim"], train_config, device,
                use_global_norm=spec["use_global_norm"],
            )

            # Evaluate on DREAM4-10
            logger.info(f"  Evaluating on DREAM4-10...")
            dream_results = evaluate_dream4_10(model, gm, gs, spec, device)
            logger.info(
                f"  DREAM4-10: AUROC={dream_results['avg_auroc']:.4f} "
                f"(±{dream_results['std_auroc']:.4f}), "
                f"AUPR={dream_results['avg_aupr']:.4f}"
            )
            for nr in dream_results["networks"]:
                logger.info(f"    {nr['network']}: AUROC={nr['auroc']:.4f} AUPR={nr['aupr']:.4f}")

            run_key = f"{variant}_{actual_scale}"
            all_results["runs"][run_key] = {
                "variant": variant,
                "scale": actual_scale,
                "dim": spec["dim"],
                "use_global_norm": spec["use_global_norm"],
                "n_train": len(train_data),
                "n_val": len(val_data),
                "best_val_loss": float(best_vl),
                "best_epoch": best_ep,
                "dream4_10": dream_results,
            }

            # Save best model per variant (by DREAM4-10 AUROC)
            if dream_results["avg_auroc"] > best_scale_auroc:
                best_scale_auroc = dream_results["avg_auroc"]
                best_scale_model = {
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": train_config,
                    "input_dim": spec["dim"],
                    "categories": spec["categories"],
                    "global_mean": gm,
                    "global_std": gs,
                    "use_global_norm": spec["use_global_norm"],
                    "best_val_loss": best_vl,
                    "best_epoch": best_ep,
                    "dream4_10": dream_results,
                    "n_train": len(train_data),
                    "scale": actual_scale,
                    "variant": variant,
                    "data_source": "gnw_dream4_aligned",
                    # Compat fields
                    "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "variant_name": "mlp",
                    "output_mode": "binary",
                    "hidden_dims": train_config["hidden_dims"],
                    "dropout": train_config["dropout"],
                    "norm": train_config["norm"],
                    "activation": train_config["activation"],
                }

            # Free memory
            del model, train_data, train_ds
            torch.cuda.empty_cache()

        # Save best model for this variant
        if best_scale_model is not None:
            model_path = OUT_DIR / f"gnw50k_{variant}_best.pt"
            torch.save(best_scale_model, model_path)
            logger.info(f"Saved best {variant} model: {model_path} (AUROC={best_scale_auroc:.4f})")

    # ── Summary ──
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY — DREAM4-10 AUROC")
    logger.info(f"{'='*60}")

    # Print scaling table
    scales_seen = sorted(set(r["scale"] for r in all_results["runs"].values()))
    variants_seen = sorted(set(r["variant"] for r in all_results["runs"].values()))

    header = f"{'Variant':<15}" + "".join(f"{s:>8}" for s in scales_seen)
    logger.info(header)
    logger.info("-" * len(header))
    for v in variants_seen:
        row = f"{v:<15}"
        for s in scales_seen:
            key = f"{v}_{s}"
            if key in all_results["runs"]:
                auroc = all_results["runs"][key]["dream4_10"]["avg_auroc"]
                row += f"{auroc:>8.4f}"
            else:
                row += f"{'N/A':>8}"
        logger.info(row)

    logger.info(f"\nBaseline (TabPFN-SCM 50K slim_deep): AUROC=0.634")
    logger.info(f"Baseline (GNW-D4aligned 8K slim): AUROC=0.548")

    # Save results
    json_path = OUT_DIR / "gnw50k_scaling_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
