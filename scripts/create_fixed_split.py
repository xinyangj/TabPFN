#!/usr/bin/env python
"""Create a fixed, deterministic train/val/test split for v6 interpretation data."""
import json, hashlib, time, sys
from pathlib import Path
import numpy as np

CACHE_DIR = Path("/home/xinyangjiang/Projects/TabPFN/data/interpretation_cache_v6_1m")
OUT_FILE = Path("/home/xinyangjiang/Projects/TabPFN/data/interpretation_splits/v6_fixed_split.json")
DIM = 1267
SALT = "tabpfn_interp_v6_fixed"  # deterministic salt

def get_split(seed_id: int) -> str:
    """Deterministic hash-based split. Stable across data additions."""
    h = hashlib.sha256(f"{SALT}_{seed_id}".encode()).hexdigest()
    bucket = int(h[:8], 16) % 12
    if bucket == 0:
        return "test"
    elif bucket == 1:
        return "val"
    else:
        return "train"

def main():
    t0 = time.time()
    files = sorted(CACHE_DIR.glob("dataset_*.npz"))
    print(f"Scanning {len(files)} npz files...")
    
    splits = {"train": [], "val": [], "test": []}
    skipped = {"bad_dim": 0, "no_labels": 0, "error": 0}
    
    for i, f in enumerate(files):
        try:
            seed_id = int(f.stem.split("_")[1])
            d = np.load(f)
            # Validate: concatenate categories and check dim
            cats = sorted([k for k in d.files if k.startswith("cat_")])
            parts = [d[k] for k in cats]
            if not parts:
                skipped["error"] += 1; continue
            fv = np.concatenate(parts, axis=1)
            if fv.shape[1] != DIM:
                skipped["bad_dim"] += 1; continue
            lab = d.get("label_binary_direct")
            if lab is None or lab.sum() == 0:
                skipped["no_labels"] += 1; continue
            
            split = get_split(seed_id)
            splits[split].append(seed_id)
        except Exception as e:
            skipped["error"] += 1
        
        if (i + 1) % 100000 == 0:
            total_valid = sum(len(v) for v in splits.values())
            print(f"  {i+1}/{len(files)}: {total_valid} valid, skip={skipped}, {time.time()-t0:.0f}s")
    
    total_valid = sum(len(v) for v in splits.values())
    total_skip = sum(skipped.values())
    print(f"\nDone in {time.time()-t0:.0f}s")
    print(f"Valid: {total_valid} ({total_skip} skipped: {skipped})")
    print(f"Split: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")
    print(f"Ratios: train={len(splits['train'])/total_valid:.1%}, val={len(splits['val'])/total_valid:.1%}, test={len(splits['test'])/total_valid:.1%}")
    
    # Sort IDs for reproducibility
    for k in splits:
        splits[k].sort()
    
    out = {
        "version": "v6",
        "dim": DIM,
        "salt": SALT,
        "split_method": "sha256_hash_mod12",
        "n_total": total_valid,
        "n_train": len(splits["train"]),
        "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "skipped": skipped,
        "train_ids": splits["train"],
        "val_ids": splits["val"],
        "test_ids": splits["test"],
    }
    
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(out, f)
    
    file_size = OUT_FILE.stat().st_size / 1e6
    print(f"Saved to {OUT_FILE} ({file_size:.1f} MB)")

if __name__ == "__main__":
    main()
