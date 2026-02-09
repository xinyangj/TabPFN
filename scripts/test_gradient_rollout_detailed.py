"""Detailed test of gradient rollout to trace execution."""

import numpy as np
import torch
from tabpfn.grn import TabPFNGRNRegressor, GRNPreprocessor
from tabpfn.grn.attention_extractor import (
    GradientAttentionExtractor,
    EdgeScoreComputer,
    compute_gradient_weighted_rollout,
    compute_sequential_attention_rollout
)
from pathlib import Path

print('Detailed gradient rollout test...')
print('=' * 70)

# Load DREAM4 data
data_path = Path('data/dream4/dream4')
X = np.load(data_path / 'dream4_10_net1_expression.npy')
genes = np.loadtxt(data_path / 'dream4_10_net1_genes.csv', delimiter=',', dtype=str)
tfs = np.loadtxt(data_path / 'dream4_10_net1_tfs.csv', delimiter=',', dtype=str)

# Use small subset
tf_subset = [tf for tf in tfs[:3] if tf in genes]
target_subset = list(genes[:1])

tf_indices = [list(genes).index(tf) for tf in tf_subset]
target_indices = [i for i, g in enumerate(genes) if g in target_subset]

X_test = X[:, tf_indices]
y_test = X[:, target_indices]

print(f'Testing with {len(tf_subset)} TFs and {len(target_subset)} targets')

# Fit gradient_rollout model
grn = TabPFNGRNRegressor(
    tf_names=tf_subset,
    target_genes=target_subset,
    n_estimators=1,
    edge_score_strategy='gradient_rollout',
    device='auto'
)
grn.fit(X_test, y_test)

# Get attention weights for the first target
target_name = target_subset[0]
attention = grn.attention_weights_[target_name]

print(f'\\nAttention weights for {target_name}:')
print(f'  Number of layers: {len(attention)}')

# Get the model
model = grn.target_models_[target_name]

# Convert to tensors
X_tensor = torch.from_numpy(X_test).float().cuda()
y_target_tensor = torch.from_numpy(y_test[:, 0]).float().cuda()

# Compute gradient head weights
gradient_extractor = GradientAttentionExtractor()
print(f'\\nComputing gradient head weights...')
head_weights = gradient_extractor.compute_gradient_head_weights(
    model=model,
    X=X_tensor,
    y_target=y_target_tensor,
    attention_weights=attention,
    normalization="l1"
)

if head_weights:
    print(f'✓ Head weights computed: {len(head_weights)} layers')
    print(f'  Sample weights from layer_0:')
    layer_0_weights = head_weights.get('layer_0', {})
    for key in ['between_features', 'between_items']:
        if key in layer_0_weights:
            print(f'    {key}: {layer_0_weights[key].cpu().numpy()}')
else:
    print('✗ No head weights computed!')
    exit(1)

# Now test the gradient-weighted rollout
print(f'\\nTesting gradient-weighted rollout...')

# Test with gradient weights
print('\\n1. Testing with gradient weights (should use gradient_weighted):')
rollout_grad = compute_gradient_weighted_rollout(
    attention_weights=attention,
    head_weights=head_weights,
    head_combination='weighted',
    add_residual=True,
    average_batch=True,
)
print(f'   Rollout matrix shape: {rollout_grad.shape}')
print(f'   Rollout matrix mean: {rollout_grad.mean().item():.6f}')
print(f'   Rollout matrix std: {rollout_grad.std().item():.6f}')

# Test without gradient weights (should fall back to sequential)
print('\\n2. Testing without gradient weights (should use sequential_rollout):')
rollout_seq = compute_gradient_weighted_rollout(
    attention_weights=attention,
    head_weights=None,  # No gradient weights
    head_combination='weighted',
    add_residual=True,
    average_batch=True,
)
print(f'   Rollout matrix shape: {rollout_seq.shape}')
print(f'   Rollout matrix mean: {rollout_seq.mean().item():.6f}')
print(f'   Rollout matrix std: {rollout_seq.std().item():.6f}')

# Test with sequential rollout directly
print('\\n3. Testing sequential rollout directly:')
rollout_seq_direct = compute_sequential_attention_rollout(
    attention_weights=attention,
    head_combination='mean',
    add_residual=True,
    average_batch=True,
)
print(f'   Rollout matrix shape: {rollout_seq_direct.shape}')
print(f'   Rollout matrix mean: {rollout_seq_direct.mean().item():.6f}')
print(f'   Rollout matrix std: {rollout_seq_direct.std().item():.6f}')

# Compare results
print('\\n' + '=' * 70)
print('COMPARISON:')
print('=' * 70)

diff_grad_seq = (rollout_grad - rollout_seq).abs().max().item()
diff_grad_seq_direct = (rollout_grad - rollout_seq_direct).abs().max().item()
diff_seq_seq_direct = (rollout_seq - rollout_seq_direct).abs().max().item()

print(f'Max difference (gradient_weighted vs no_weights): {diff_grad_seq:.8f}')
print(f'Max difference (gradient_weighted vs sequential): {diff_grad_seq_direct:.8f}')
print(f'Max difference (no_weights vs sequential): {diff_seq_seq_direct:.8f}')

if diff_grad_seq > 0.001:
    print('\\n✓ Gradient weighting produces DIFFERENT results!')
else:
    print('\\n⚠ Gradient weighting produces SAME results (unexpected)')
