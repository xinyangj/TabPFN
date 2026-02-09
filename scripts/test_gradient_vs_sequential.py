"""Test gradient rollout vs sequential rollout on DREAM4 data."""

import numpy as np
import torch
from tabpfn.grn import TabPFNGRNRegressor, GRNPreprocessor
from pathlib import Path
import pandas as pd

print('Testing gradient_rollout vs sequential_rollout on DREAM4 data...')
print('=' * 70)

# Load DREAM4 data
data_path = Path('data/dream4/dream4')

# Load 10-gene network data
X = np.load(data_path / 'dream4_10_net1_expression.npy')
genes = np.loadtxt(data_path / 'dream4_10_net1_genes.csv', delimiter=',', dtype=str)
tfs = np.loadtxt(data_path / 'dream4_10_net1_tfs.csv', delimiter=',', dtype=str)
ground_truth_df = pd.read_csv(data_path / 'dream4_10_net1_gold_standard.csv')

print(f'Data: X shape {X.shape}')
print(f'Genes: {len(genes)}, TFs: {len(tfs)}')
print(f'Ground truth edges: {len(ground_truth_df)}')
print()

# Use subset for quick test
# Make sure we only use TFs that are also in genes
n_tfs = min(5, len(tfs))
n_targets = min(3, len(genes))
tf_subset = [tf for tf in tfs[:n_tfs] if tf in genes]  # Ensure TFs are in genes
target_subset = list(genes[:n_targets])

# Get indices
tf_indices = [list(genes).index(tf) for tf in tf_subset]
target_indices = [i for i, g in enumerate(genes) if g in target_subset]

X_test = X[:, tf_indices]
y_test = X[:, target_indices]

print(f'Testing with {len(tf_subset)} TFs and {len(target_subset)} targets')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
print()

# Run sequential_rollout
print('Running sequential_rollout...')
grn_seq = TabPFNGRNRegressor(
    tf_names=tf_subset,
    target_genes=target_subset,
    n_estimators=1,
    edge_score_strategy='sequential_rollout',
    device='auto'
)
grn_seq.fit(X_test, y_test)
scores_seq = grn_seq.get_edge_scores()
print(f'  Sequential rollout edges: {len(scores_seq)}')

# Run gradient_rollout
print()
print('Running gradient_rollout...')
grn_grad = TabPFNGRNRegressor(
    tf_names=tf_subset,
    target_genes=target_subset,
    n_estimators=1,
    edge_score_strategy='gradient_rollout',
    device='auto'
)
grn_grad.fit(X_test, y_test)
scores_grad = grn_grad.get_edge_scores()
print(f'  Gradient rollout edges: {len(scores_grad)}')

# Compare results
print()
print('Comparing edge scores (all edges):')
all_edges = sorted(set(scores_seq.keys()) | set(scores_grad.keys()))

max_diff = 0
diff_edges = 0
for edge in all_edges:
    seq_score = scores_seq.get(edge, 0.0)
    grad_score = scores_grad.get(edge, 0.0)
    diff = abs(seq_score - grad_score)
    max_diff = max(max_diff, diff)
    if diff > 0.01:
        diff_edges += 1
        print(f'  {edge}: Sequential={seq_score:.4f}, Gradient={grad_score:.4f}, Diff={diff:.4f}')

print()
print(f'Maximum difference: {max_diff:.4f}')
print(f'Edges with difference > 0.01: {diff_edges}/{len(all_edges)}')

if max_diff > 0.001:
    print('✓ Gradient rollout produces DIFFERENT results!')
    print('  Gradient-based head importance weighting is working!')
else:
    print('⚠ Results are very similar')
