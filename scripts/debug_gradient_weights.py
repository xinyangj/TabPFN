"""Debug gradient weight application in rollout."""

import numpy as np
import torch
from tabpfn.grn import TabPFNGRNRegressor
from pathlib import Path

print('Debugging gradient weight application...')
print('=' * 70)

# Load DREAM4 data
data_path = Path('data/dream4/dream4')

# Load 10-gene network data
X = np.load(data_path / 'dream4_10_net1_expression.npy')
genes = np.loadtxt(data_path / 'dream4_10_net1_genes.csv', delimiter=',', dtype=str)
tfs = np.loadtxt(data_path / 'dream4_10_net1_tfs.csv', delimiter=',', dtype=str)

# Use subset for quick test
tf_subset = [tf for tf in tfs[:4] if tf in genes]
target_subset = list(genes[:2])

tf_indices = [list(genes).index(tf) for tf in tf_subset]
target_indices = [i for i, g in enumerate(genes) if g in target_subset]

X_test = X[:, tf_indices]
y_test = X[:, target_indices]

print(f'Testing with {len(tf_subset)} TFs and {len(target_subset)} targets')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
print()

# Run gradient_rollout to get attention weights and head weights
print('Running gradient_rollout to extract attention weights...')
grn_grad = TabPFNGRNRegressor(
    tf_names=tf_subset,
    target_genes=target_subset,
    n_estimators=1,
    edge_score_strategy='gradient_rollout',
    device='auto'
)
grn_grad.fit(X_test, y_test)

# Get attention weights for one target
target_name = target_subset[0]
attention = grn_grad.attention_weights_[target_name]

print(f'\\nAttention weights for {target_name}:')
print(f'  Layers: {len(attention)}')

# Show layer 0 details
layer_0 = attention['layer_0']
feat_attn = layer_0['between_features']  # (batch, n_items, n_items, nheads)
item_attn = layer_0['between_items']     # (batch, n_fblocks, n_fblocks, nheads)

print(f'\\nLayer 0 between_features:')
print(f'  Shape: {feat_attn.shape}')
print(f'  Mean values per head (before weighting):')
for h in range(feat_attn.size(-1)):
    print(f'    Head {h}: {feat_attn.mean(dim=(0,1,2))[h].item():.6f}')

print(f'\\nLayer 0 between_items:')
print(f'  Shape: {item_attn.shape}')
print(f'  Mean values per head (before weighting):')
for h in range(item_attn.size(-1)):
    print(f'    Head {h}: {item_attn.mean(dim=(0,1,2))[h].item():.6f}')

# Now let's manually compute gradient-based rollout and compare
from tabpfn.grn.attention_extractor import GradientAttentionExtractor, EdgeScoreComputer

gradient_extractor = GradientAttentionExtractor()

# Get the model for this target
model = grn_grad.target_models_[target_name]

# Convert to tensors
X_tensor = torch.from_numpy(X_test).float().cuda()
y_target_tensor = torch.from_numpy(y_test[:, 0]).float().cuda()  # First target

# Compute gradient head weights
print(f'\\nComputing gradient head weights for {target_name}...')
head_weights = gradient_extractor.compute_gradient_head_weights(
    model=model,
    X=X_tensor,
    y_target=y_target_tensor,
    attention_weights=attention,
    normalization="l1"
)

if head_weights:
    print(f'\\nGradient head weights:')
    layer_0_weights = head_weights.get('layer_0', {})
    if 'between_features' in layer_0_weights:
        w_feat = layer_0_weights['between_features']
        print(f'  between_features: {w_feat.cpu().numpy()}')
    if 'between_items' in layer_0_weights:
        w_items = layer_0_weights['between_items']
        print(f'  between_items: {w_items.cpu().numpy()}')

    # Now apply weights manually and compare
    print(f'\\nLayer 0 between_features after gradient weighting:')
    feat_attn_avg = feat_attn.mean(dim=0)  # (n_items, n_items, nheads)
    if 'between_features' in layer_0_weights:
        w_feat = layer_0_weights['between_features']
        feat_attn_weighted = feat_attn_avg * w_feat.view(1, 1, -1)
        print(f'  Mean values per head (after weighting):')
        for h in range(feat_attn_weighted.size(-1)):
            orig = feat_attn_avg[:, :, h].mean().item()
            weighted = feat_attn_weighted[:, :, h].mean().item()
            diff = abs(weighted - orig)
            print(f'    Head {h}: {weighted:.6f} (diff from orig: {diff:+.6f})')

    print(f'\\nLayer 0 between_items after gradient weighting:')
    item_attn_avg = item_attn.mean(dim=0)  # (n_fblocks, n_fblocks, nheads)
    if 'between_items' in layer_0_weights:
        w_items = layer_0_weights['between_items']
        item_attn_weighted = item_attn_avg * w_items.view(1, 1, -1)
        print(f'  Mean values per head (after weighting):')
        for h in range(item_attn_weighted.size(-1)):
            orig = item_attn_avg[:, :, h].mean().item()
            weighted = item_attn_weighted[:, :, h].mean().item()
            diff = abs(weighted - orig)
            print(f'    Head {h}: {weighted:.6f} (diff from orig: {diff:+.6f})')
else:
    print('No gradient weights computed!')

print(f'\\n' + '=' * 70)
