#!/usr/bin/env python3
"""Preprocess DREAM challenge datasets from NSRGRN format to TabPFN format.

This script converts DREAM4 datasets from the NSRGRN repository format
to the format expected by TabPFN's DREAMChallengeLoader.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def preprocess_dream4_network(
    data_dir: Path,
    gold_dir: Path,
    output_dir: Path,
    network_size: int,
    network_id: int
):
    """Preprocess a single DREAM4 network.

    Parameters
    ----------
    data_dir : Path
        Directory containing expression data (DREAM4_InSilico_Size10, etc.)
    gold_dir : Path
        Directory containing gold standard networks
    output_dir : Path
        Directory to save processed files
    network_size : int
        Size of network (10, 50, 100)
    network_id : int
        Network number (1-5)
    """
    # Prefix for output files
    prefix = f"dream4_{network_size}_net{network_id}"

    # Determine the expression file to use
    network_subdir = data_dir / f"insilico_size{network_size}_{network_id}"

    if not network_subdir.exists():
        print(f"Warning: Network directory not found: {network_subdir}")
        return False

    # Stack ALL available expression data types to match GNW training composition.
    # Order: wildtype, knockouts, knockdowns, multifactorial, timeseries
    data_types = ["wildtype", "knockouts", "knockdowns", "multifactorial", "timeseries"]
    prefix_stem = f"insilico_size{network_size}_{network_id}"

    gene_names = None
    data_parts = []
    loaded_types = []

    for dtype in data_types:
        fpath = network_subdir / f"{prefix_stem}_{dtype}.tsv"
        if not fpath.exists():
            continue
        df = pd.read_csv(fpath, sep="\t")
        if gene_names is None:
            gene_names = [col for col in df.columns if col != "Time"]
        expr = df.drop(columns=["Time"], errors="ignore").values.astype(np.float32)
        data_parts.append(expr)
        loaded_types.append(f"{dtype}({expr.shape[0]})")

    if not data_parts:
        print(f"Warning: No expression file found for network {network_id}")
        return False

    expression = np.vstack(data_parts)
    print(f"  Stacked {' + '.join(loaded_types)} = {expression.shape[0]} samples")
    print(f"  Expression shape: {expression.shape} (samples x genes)")
    print(f"  Number of genes: {len(gene_names)}")
    
    # 2. Load gold standard
    gold_file = gold_dir / f"DREAM4_GoldStandard_InSilico_Size{network_size}_{network_id}.tsv"
    if not gold_file.exists():
        print(f"Warning: Gold standard file not found: {gold_file}")
        return False
    
    gold_df = pd.read_csv(gold_file, sep='\t', header=None, names=['tf', 'target', 'weight'])
    
    # 3. Determine TFs and targets
    # In DREAM4, all genes can be both TFs and targets
    # Use all genes that appear as TFs in the gold standard
    gold_tfs = set(gold_df['tf'].unique())
    tf_genes = sorted([g for g in gene_names if g in gold_tfs])
    target_genes = sorted(gene_names)  # All genes can be targets

    print(f"  Number of TFs: {len(tf_genes)}")
    print(f"  Number of potential targets: {len(target_genes)}")
    print(f"  Number of gold edges: {len(gold_df)}")
    print(f"  Gold edges with TF in our list: {len(gold_df[gold_df['tf'].isin(tf_genes)])}")

    # 4. Save files in TabPFN format
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save expression
    np.save(output_dir / f"{prefix}_expression.npy", expression)
    
    # Save gene names
    pd.DataFrame({'gene': gene_names}).to_csv(
        output_dir / f"{prefix}_genes.csv", index=False
    )
    
    # Save TF names
    pd.DataFrame({'tf': tf_genes}).to_csv(
        output_dir / f"{prefix}_tfs.csv", index=False
    )
    
    # Save gold standard
    gold_df.to_csv(output_dir / f"{prefix}_gold_standard.csv", index=False)
    
    print(f"  Saved to: {output_dir / prefix}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Preprocess DREAM datasets')
    parser.add_argument('--input-dir', type=str, default='data/dream4/dream4/NSRGRN-main',
                       help='Input directory with NSRGRN data')
    parser.add_argument('--output-dir', type=str, default='data/dream4/dream4',
                       help='Output directory for processed data')
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Process all network sizes and IDs
    for size in [10, 50, 100]:
        data_dir = input_dir / f"DREAM4_InSilico_Size{size}"
        gold_dir = input_dir / "DREAM4_InSilicoNetworks_GoldStandard" / f"Size {size}"

        if not data_dir.exists():
            print(f"Warning: Data directory not found: {data_dir}")
            continue

        if not gold_dir.exists():
            print(f"Warning: Gold directory not found: {gold_dir}")
            continue

        print(f"\n{'='*70}")
        print(f"Processing DREAM4 {size}-gene networks")
        print(f"{'='*70}")

        for network_id in range(1, 6):
            success = preprocess_dream4_network(
                data_dir, gold_dir, output_dir, size, network_id
            )
            if success:
                print(f"  ✓ Processed network {network_id}")
            else:
                print(f"  ✗ Failed to process network {network_id}")

if __name__ == "__main__":
    main()
