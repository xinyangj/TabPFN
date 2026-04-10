# TabPFN Attention: Code Examples and Implementation Details

## Example 1: Understanding Feature Grouping

```python
import numpy as np
from tabpfn import TabPFNRegressor

# Setup: 7 features, features_per_group=2 (default)
X_train = np.random.randn(100, 7)
y_train = np.random.randn(100)
X_test = np.random.randn(20, 7)

# Step 1: Feature grouping in transformer forward pass
# From transformer.py:369-395
features_per_group = 2
n_features = 7

# Pad to multiple of features_per_group
missing_to_next = (features_per_group - (n_features % features_per_group)) % features_per_group
if missing_to_next > 0:
    X_train = np.pad(X_train, ((0,0), (0, missing_to_next)))
    # Now shape: (100, 8)

# Reshape into blocks
# Original: (seq_len=100, batch=1, n_features=8)
# After einops.rearrange(..., "s b (f n) -> b s f n", n=2)
# Result: (batch=1, seq_len=100, n_feature_blocks=4, features_per_group=2)

n_feature_blocks = 8 // 2  # = 4
n_blocks = n_feature_blocks + 1  # = 5 (add target block)

print(f"Features: 7 → Padded: 8 → Blocks: {n_feature_blocks} + 1 target = {n_blocks}")
# Output:
# Features: 7 → Padded: 8 → Blocks: 4 + 1 target = 5
```

## Example 2: Extracting Attention Weights

```python
from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor

# Train your model
model = TabPFNRegressor(n_estimators=1)
model.fit(X_train, y_train)

# Extract signals with full gradient mode
extractor = SignalExtractor(extract_gradients=True)
signals = extractor.extract(model, X_train, y_train, X_test)

# Inspect attention weights
between_features = signals['between_features_attention']
between_items = signals['between_items_attention']

print(f"Number of layers: {len(between_features)}")
# Output: Number of layers: 12

# Between-features attention for layer 0
attn_feat_l0 = between_features['layer_0']
print(f"Between-features shape: {attn_feat_l0.shape}")
# Output: Between-features shape: torch.Size([1, 5, 5, 6])
#         (batch=1, n_blocks=5, n_blocks=5, n_heads=6)

# Between-items attention for layer 0
attn_items_l0 = between_items['layer_0']
print(f"Between-items shape: {attn_items_l0.shape}")
# Output: Between-items shape: torch.Size([5, 120, 120, 6])
#         (n_blocks=5, n_items=120, n_items=120, n_heads=6)
#         n_items = n_train + n_test = 100 + 20 = 120
```

## Example 3: Understanding Between-Features Attention

```python
# From signal_processor.py:338-478
import torch
import numpy as np

# Simulated layer attention: (batch=1, n_blocks=5, n_blocks=5, n_heads=6)
attn = torch.randn(1, 5, 5, 6).softmax(dim=2)  # Softmax over key dimension

# Mean over batch
mean_attn = attn.mean(dim=0)  # (5, 5, 6)

# Extract stats for each feature block
n_feature_blocks = 4
target_block = 4

# For head 0
h = 0
ah = mean_attn[:, :, h]  # (5, 5) attention matrix
feat_ah = ah[:n_feature_blocks, :]  # (4, 5) feature blocks attending to all

# Compute basic stats for feature block 0
fb = 0

# 1. Self-attention (diagonal)
self_attn = ah[fb, fb].item()
print(f"Block {fb} self-attention: {self_attn:.4f}")

# 2. Attention to target
to_target = feat_ah[fb, target_block].item()
print(f"Block {fb} → target: {to_target:.4f}")

# 3. Attention from target
from_target = ah[target_block, fb].item()
print(f"Target → block {fb}: {from_target:.4f}")

# 4. Mean attention to other feature blocks (excluding self)
if n_feature_blocks > 1:
    out_to_feats = feat_ah[:, :n_feature_blocks]  # (4, 4)
    row_sum = out_to_feats.sum(dim=1)[fb].item()
    self_val = out_to_feats[fb, fb].item()
    mean_to_others = (row_sum - self_val) / (n_feature_blocks - 1)
    print(f"Block {fb} → others (mean): {mean_to_others:.4f}")

# 5. Mean attention from other feature blocks (excluding self)
col_sum = out_to_feats.sum(dim=0)[fb].item()
mean_from_others = (col_sum - self_val) / (n_feature_blocks - 1)
print(f"Others → block {fb} (mean): {mean_from_others:.4f}")

# 6. Entropy
a_clamped = torch.clamp(ah[:n_feature_blocks, :], min=1e-10)
entropy = -(a_clamped * torch.log(a_clamped)).sum(dim=1)[fb].item()
print(f"Block {fb} entropy: {entropy:.4f}")
```

## Example 4: Block → Feature Mapping

```python
# From signal_processor.py:372-373
import numpy as np

n_features = 7
n_feature_blocks = 4  # ceil(7 / 2)

# Create block index array
# Maps each feature to its corresponding block
bi_arr = np.array([
    min(i * n_feature_blocks // n_features, n_feature_blocks - 1)
    for i in range(n_features)
])

print("Feature → Block mapping:")
for i, block_idx in enumerate(bi_arr):
    print(f"  Feature {i} → Block {block_idx}")

# Output:
# Feature → Block mapping:
#   Feature 0 → Block 0
#   Feature 1 → Block 0
#   Feature 2 → Block 1
#   Feature 3 → Block 1
#   Feature 4 → Block 2
#   Feature 5 → Block 2
#   Feature 6 → Block 3

# Now gather block stats to per-feature level
# block_stats shape: (n_layers=12, n_feature_blocks=4, n_heads=6, n_stats=6)
block_stats = np.random.randn(12, 4, 6, 6)

# Gather: (12, 4, 6, 6) → (12, 7, 6, 6)
gathered = block_stats[:, bi_arr, :, :]
print(f"Gathered shape: {gathered.shape}")  # (12, 7, 6, 6)

# Now each feature has its own stats vector
# Reshape for per-feature vector
per_feature_vector = gathered.transpose(1, 0, 2, 3).reshape(7, -1)
print(f"Per-feature vector shape: {per_feature_vector.shape}")  # (7, 432)
```

## Example 5: GPU Stats Computer Fast Path

```python
from tabpfn.interpretation.extraction.gpu_stats_computer import GPUStatsComputer
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
import torch

# Initialize
gpu_computer = GPUStatsComputer(enriched=False)

# Extract signals (signals have GPU tensors)
extractor = SignalExtractor(extract_gradients=False)
signals = extractor.extract(model, X_train, y_train, X_test)

# Compute stats on GPU (fast!)
gpu_stats = gpu_computer.compute(signals)

# Now process them on CPU (quick because stats are tiny)
processor = SignalProcessor(enriched=False)
feature_vectors = processor.process_from_stats(gpu_stats, n_features=7)

print(f"Feature vectors shape: {feature_vectors.shape}")  # (7, 1300) approx

# Example of what's in gpu_stats:
print("GPU stats keys:", gpu_stats.keys())
# Output:
# dict_keys(['items_attention_block_stats', 'features_attention_block_stats',
#            'features_to_target_per_layer', 'mlp_block_stats', 'test_embeddings',
#            'train_embeddings', 'input_gradients'])

print(f"Features attn block stats shape: {gpu_stats['features_attention_block_stats'].shape}")
# Output: (12, 4, 6, 6)  → 12 layers, 4 feature blocks, 6 heads, 6 stats
```

## Example 6: Straight-Through Estimator for Gradient Computation

```python
# From full_attention.py:749-764

import torch
import torch.nn.functional as F

# Simplified SDPA forward + manual backward
batch_size, seq_len, nhead, d_k = 1, 5, 6, 32
d_v = 32

# Example Q, K, V
Q = torch.randn(batch_size, seq_len, nhead, d_k, requires_grad=True)
K = torch.randn(batch_size, seq_len, nhead, d_k)
V = torch.randn(batch_size, seq_len, nhead, d_v)

# SDPA forward (fused, fast)
sdpa_output = F.scaled_dot_product_attention(
    Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
)
sdpa_output = sdpa_output.transpose(1, 2)

# Manual shadow computation (for gradients)
scale = 1.0 / (d_k ** 0.5)
logits = torch.einsum("b q h d, b k h d -> b q k h", Q, K)
logits = logits * scale
attention_weights = torch.softmax(logits, dim=2)
shadow_output = torch.einsum("b q k h, b k h d -> b q h d", attention_weights, V)

# Straight-through estimator: use shadow in backward, SDPA in forward
attention_head_outputs = shadow_output + (sdpa_output - shadow_output).detach()

# Backward: gradients flow through shadow (manual attention)
loss = attention_head_outputs.sum()
loss.backward()

print(f"Q.grad is not None: {Q.grad is not None}")  # True
print(f"Gradient flows through shadow attention computation")
```

## Example 7: Enriched vs Basic Mode Statistics

```python
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

# Extract signals
signals = extractor.extract(model, X_train, y_train, X_test)

# Basic mode
processor_basic = SignalProcessor(enriched=False)
vectors_basic = processor_basic.process(signals)
print(f"Basic mode shape: {vectors_basic.shape}")
# Output: (7, ~1300) with features_per_group=2

# Enriched mode (get more detailed statistics)
processor_enriched = SignalProcessor(enriched=True)
vectors_enriched = processor_enriched.process(signals)
print(f"Enriched mode shape: {vectors_enriched.shape}")
# Output: (7, ~2400)

# Enriched mode adds:
# - Between-features: 6→15 stats/head (9 additional)
# - Between-items: 3→9 stats/head (6 additional if train/test split)
# That's 12 layers × 6 heads × 9 additional = 648 extra dims for between-features
#     and 12 × 6 × 6 = 432 extra dims for between-items
```

## Example 8: Extracting Train vs Test Embeddings

```python
signals = extractor.extract(model, X_train, y_train, X_test)

# Get embeddings at final layer
train_emb = signals['train_embeddings']  # (n_train, emsize=192)
test_emb = signals['test_embeddings']    # (n_test, emsize=192)

print(f"Train embeddings shape: {train_emb.shape}")  # (100, 192)
print(f"Test embeddings shape: {test_emb.shape}")    # (20, 192)

# These are the final encoder outputs before the decoder head
# Useful for analyzing what the model "thinks" about each sample

# In signal processor, these are aggregated as global context
# and replicated for each feature (since embeddings are sample-level, not feature-level)
```

## Example 9: Between-Items Attention with Train/Test Split

```python
# From signal_processor.py:540-551
import torch
import numpy as np

# Simulated between-items attention
# (n_blocks=5, n_items=120, n_items=120, n_heads=6)
# where n_items = n_train(100) + n_test(20)
attn = torch.randn(5, 120, 120, 6).softmax(dim=2)

n_train = 100
n_test = 20
n_blocks = 5
n_heads = 6

# Extract enriched stats for block 0, head 0
block = 0
head = 0
ah = attn[block, :, :, head]  # (120, 120)

# Split attention matrix into regions
t2t = ah[:n_train, n_train:]    # train→test (100, 20)
t2s = ah[n_train:, :n_train]    # test→train (20, 100)
st = ah[:n_train, :n_train]     # self-train (100, 100)
ss = ah[n_train:, n_train:]     # self-test (20, 20)

# Compute enriched stats
mean_t2t = t2t.mean().item()  # Train attends to test
mean_t2s = t2s.mean().item()  # Test attends to train
mean_st = st.mean().item()    # Trains attend to each other
mean_ss = ss.mean().item()    # Tests attend to each other

print(f"Train → Test: {mean_t2t:.4f}")
print(f"Test → Train: {mean_t2s:.4f}")
print(f"Train ↔ Train: {mean_st:.4f}")
print(f"Test ↔ Test: {mean_ss:.4f}")

# Ratio metric: are training samples using test samples?
ratio = mean_t2t / (mean_st + 1e-10)
print(f"Train→Test / Train↔Train ratio: {ratio:.4f}")

# Concentration: how focused is attention on one item?
row_sum = ah.sum(dim=1)  # (120,)
max_attn = ah.max().item()
concentration = max_attn / (row_sum.mean().item() + 1e-10)
print(f"Concentration metric: {concentration:.4f}")
```

## Example 10: Hook Registration for MLP Activations

```python
from torch import nn

# From signal_extractor.py:187-203

mlp_activations = {}

def register_mlp_hooks(layers):
    hooks = []
    for i, layer in enumerate(layers):
        def hook_fn(module, input, output, layer_idx=i):
            # Store MLP output on the fly
            mlp_activations[layer_idx] = output.detach()
        
        hooks.append(layer.mlp.register_forward_hook(hook_fn))
    return hooks

# During forward pass, all MLP outputs are captured
# After forward: mlp_activations contains:
# {
#     0: torch.Tensor,  # (batch, batch_items, n_blocks, emsize)
#     1: torch.Tensor,
#     ...
#     11: torch.Tensor  # 12 layers total
# }

# These are then processed to extract per-block statistics:
# - mean: average activation per block
# - std: activation variance per block
# - max: peak activation per block
# - sparsity: fraction of near-zero activations
# - l2_norm: L2 norm of activation vector
```

## Example 11: Complete Pipeline: Extract → Process → Use

```python
from tabpfn import TabPFNRegressor
from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
import numpy as np

# 1. Train TabPFN
X_train = np.random.randn(100, 7)
y_train = np.random.randn(100)
X_test = np.random.randn(20, 7)

model = TabPFNRegressor(n_estimators=1)
model.fit(X_train, y_train)

# 2. Extract signals (different modes available)
for mode_name, extract_grad_setting in [
    ("Full gradients", True),
    ("Input gradients only", "input_only"),
    ("No gradients", False)
]:
    print(f"\n=== {mode_name} ===")
    
    extractor = SignalExtractor(extract_gradients=extract_grad_setting)
    signals = extractor.extract(model, X_train, y_train, X_test)
    
    # 3. Process signals into feature vectors
    processor = SignalProcessor(enriched=True)
    feature_vectors = processor.process(signals)
    
    print(f"Feature vectors shape: {feature_vectors.shape}")
    print(f"Dimensions per feature: {feature_vectors.shape[1]}")
    print(f"Memory: {feature_vectors.nbytes / 1e6:.1f} MB")
    
    # These feature vectors can now be used as input to an interpretation model
    # that explains feature importance or interactions
```

## Example 12: Memory Optimization with GPU Stats Computer

```python
from tabpfn.interpretation.extraction.gpu_stats_computer import GPUStatsComputer
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
import sys

# Without GPU stats: raw attention tensors are ~1 GB
# (batch, n_blocks, n_blocks, n_heads) for each of 12 layers

print("=== Direct Processing (Large Memory) ===")
extractor = SignalExtractor(extract_gradients=False)
signals = extractor.extract(model, X_train, y_train, X_test)

# Rough memory estimate
between_feat_attn = signals['between_features_attention']['layer_0']
raw_memory_per_layer = between_feat_attn.element_size() * between_feat_attn.numel()
total_raw_memory = raw_memory_per_layer * 12
print(f"Raw attention memory (12 layers): {total_raw_memory / 1e9:.2f} GB")

print("\n=== GPU Stats Computer (Memory Efficient) ===")
gpu_computer = GPUStatsComputer(enriched=False)
gpu_stats = gpu_computer.compute(signals)

# GPU stats are tiny
feat_stats = gpu_stats['features_attention_block_stats']
gpu_memory = feat_stats.nbytes + gpu_stats['features_to_target_per_layer'].nbytes
for key in ['items_attention_block_stats', 'mlp_block_stats']:
    if key in gpu_stats:
        gpu_memory += gpu_stats[key].nbytes
print(f"GPU stats memory: {gpu_memory / 1e3:.1f} KB")

processor = SignalProcessor(enriched=False)
feature_vectors = processor.process_from_stats(gpu_stats, n_features=7)
print(f"Final feature vector memory: {feature_vectors.nbytes / 1e6:.1f} MB")
print(f"Compression ratio: {total_raw_memory / feature_vectors.nbytes:.0f}x")
```

