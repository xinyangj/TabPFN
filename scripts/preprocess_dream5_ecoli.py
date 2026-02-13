#!/usr/bin/env python3
"""Preprocess DREAM5 E. coli dataset from AGRN format to TabPFN format."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def preprocess_dream5_ecoli(agrn_dir: Path, output_dir: Path):
    """Preprocess DREAM5 E. coli dataset.

    Parameters
    ----------
    agrn_dir : Path
        Path to AGRN Dataset directory
    output_dir : Path
        Path to save processed files
    """
    # Paths to input files
    expr_file = agrn_dir / "Dataset/DREAM5/training data/Network 3 - E. coli/net3_expression_data.tsv"
    tf_file = agrn_dir / "Dataset/DREAM5/training data/Network 3 - E. coli/net3_transcription_factors.tsv"
    gene_file = agrn_dir / "Dataset/DREAM5/training data/Network 3 - E. coli/net3_gene_ids.tsv"
    gold_file = agrn_dir / "Dataset/DREAM5/test data/DREAM5_NetworkInference_GoldStandard_Network3 - E. coli.tsv"

    print(f"Loading DREAM5 E. coli data...")

    # Load expression data
    print(f"  Loading expression data from {expr_file}...")
    expr_df = pd.read_csv(expr_file, sep='\t')
    # Expression has genes as columns, samples as rows
    expression = expr_df.values.astype(np.float32)
    gene_names = list(expr_df.columns)

    print(f"  Expression shape: {expression.shape} (samples x genes)")
    print(f"  Number of genes: {len(gene_names)}")

    # Load TF names
    print(f"  Loading TF names from {tf_file}...")
    tf_df = pd.read_csv(tf_file, sep='\t', header=None)
    tf_names = tf_df[0].tolist()

    # Ensure TF names are in gene names
    tf_names = [tf for tf in tf_names if tf in gene_names]
    print(f"  Number of TFs: {len(tf_names)}")

    # Load gold standard
    print(f"  Loading gold standard from {gold_file}...")
    gold_df = pd.read_csv(gold_file, sep='\t')
    print(f"  Number of gold edges: {len(gold_df)}")

    # Save files in TabPFN format
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save expression (transpose to genes x samples format expected by loader)
    # Actually, for TabPFN we need samples x genes format
    np.save(output_dir / "ecoli_expression.npy", expression)

    # Save gene names
    pd.DataFrame({'gene': gene_names}).to_csv(
        output_dir / "ecoli_genes.csv", index=False
    )

    # Save TF names
    pd.DataFrame({'tf': tf_names}).to_csv(
        output_dir / "ecoli_tfs.csv", index=False
    )

    # Save gold standard
    gold_df.to_csv(output_dir / "ecoli_gold_standard.csv", index=False)

    print(f"  Saved to: {output_dir}")
    print(f"  ✓ DREAM5 E. coli preprocessing complete")

def main():
    parser = argparse.ArgumentParser(description='Preprocess DREAM5 E. coli dataset')
    parser.add_argument('--input-dir', type=str, default='data/dream4/dream4/AGRN',
                       help='Path to AGRN directory')
    parser.add_argument('--output-dir', type=str, default='data/dream5',
                       help='Path to save processed files')

    args = parser.parse_args()

    agrn_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    preprocess_dream5_ecoli(agrn_dir, output_dir)

if __name__ == "__main__":
    main()
