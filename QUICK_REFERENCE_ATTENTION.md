# TabPFN Attention Architecture - Quick Reference

## 🔑 KEY INSIGHTS

### Feature Grouping
- **Features are grouped** into "blocks" of size `features_per_group` (default: 2)
- **Target variable** becomes a separate block (always last): `n_blocks = ceil(n_features/2) + 1`
- Example: 7 features → [block0: f0-1], [block1: f2-3], [block2: f4-5], [block3: f6], [block4: target]

### Model Configuration (Defaults)
- `emsize = 192` (embedding dimension)
- `nhead = 6` (attention heads)
- `d_k = d_v = 32` (per-head dimension)
- `nlayers = 12` (transformer layers)

---

## 📊 ATTENTION SHAPES AT A GLANCE

### Between-Features Attention (intra-sample feature dependencies)
```
Shape: (batch_items, n_blocks, n_blocks, n_heads)

Example with 7 features, features_per_group=2:
  n_blocks = 5 (4 feature blocks + 1 target)
  Shape = (batch, 5, 5, 6)

Semantic: [i, j] = how much feature block i attends to feature block j
```

### Between-Items Attention (inter-sample relationships)
```
Shape: (n_blocks, n_items, n_items, n_heads)

Example with 20 train + 5 test, 7 features:
  n_blocks = 5
  n_items = 25
  Shape = (5, 25, 25, 6)

Semantic: [i, j, k] = attention of item j in block i to item k in block i
```

---

## 🔄 SIGNAL EXTRACTION FLOW

### Three Gradient Modes

| Mode | Gradients | Speed | Dimensions Lost |
|------|-----------|-------|-----------------|
| `True` | Full (att weights + input) | Slowest | 0 |
| `"input_only"` | Input only (no att weights) | Medium | 432/1591 |
| `False` | None | Fastest | 436 |

### Hook Mechanisms
```python
# Attention weights are cached within forward pass:
# signal_extractor.py:168-176 enables:
layer.self_attn_between_features.enable_attention_weights_return(True)
layer.self_attn_between_items.enable_attention_weights_return(True)

# And retrieved via:
layer.self_attn_between_features.get_attention_weights()  # (batch, n_blocks, n_blocks, n_heads)
layer.self_attn_between_items.get_attention_weights()     # (n_blocks, n_items, n_items, n_heads)
```

---

## 📈 SIGNAL PROCESSING PIPELINE

### Per-Feature Statistics Extraction

#### Between-Features (signal_processor.py:338-478)
**Basic Mode (6 stats/head × 12 layers × 6 heads = 432 dims):**
1. Self-attention (diagonal)
2. Attention to target
3. Attention from target
4. Mean attention to other feature blocks
5. Mean attention from other feature blocks
6. Entropy

**Enriched Mode** (+9 additional stats):
- Max/std to/from others
- Asymmetry metrics
- Target-relative ranking
- Contrast to target

#### Between-Items (signal_processor.py:480-555)
**Basic Mode (3 stats/head × 12 layers × 6 heads = 216 dims):**
1. Entropy (row-wise)
2. Max attention
3. Variance

**Enriched Mode** (+6 additional stats, if train/test split exists):
- Train→test / test→train mean attention
- Self-train / self-test mean attention
- Train/test attention ratio
- Concentration metric

### Block→Feature Mapping
```python
# Map each of n_features to its corresponding block
n_feature_blocks = n_blocks - 1  # Exclude target
bi_arr = [min(i * n_feature_blocks // n_features, n_feature_blocks - 1) 
          for i in range(n_features)]

# Gather stats: (n_layers, n_feature_blocks, n_heads, n_stats) 
#            → (n_layers, n_features, n_heads, n_stats)
gathered = block_stats[:, bi_arr, :, :]
```

---

## 🚀 GPU STATS COMPUTER (Fast Path)

**Purpose:** Compute all stats on GPU before transfer (~1 GB raw → ~20 KB stats)

### Computed Tensors
```
Between-features: (n_layers, n_feature_blocks, n_heads, 6) 
                  + (n_layers, n_feature_blocks) for cross-layer stats

Between-items:    (n_layers, n_blocks, n_heads, 3)

MLP activations:  (n_layers, n_feature_blocks, 5)
                  [mean, std, max, sparsity, l2_norm]
```

### Key GPU Operations (full_attention.py:752-772)
```python
# Attention weights computed as:
logits = einsum("b q h d, b k h d -> b q k h", Q, K)
logits = logits * (1 / sqrt(d_k))
attention_weights = softmax(logits, dim=2)

# Straight-through SDPA (with retain_grad):
shadow_output = einsum("b q k h, b k h d -> b q h d", attention_weights, V)
attention_head_outputs = shadow_output + (sdpa_output - shadow_output).detach()
```

---

## 📁 CRITICAL CODE LOCATIONS

| What | Where | Shape Output |
|------|-------|--------------|
| Attention caching | `full_attention.py:565-571` | Stored in `_cached_attention_weights` |
| Feature extraction | `signal_extractor.py:165-176` | `get_attention_weights()` |
| Between-feat stats | `signal_processor.py:338` | `(n_features, 432)` basic / `(n_features, 1083)` enriched |
| Between-items stats | `signal_processor.py:480` | `(n_features, 216)` basic / `(n_features, 648)` enriched |
| GPU stats (feat) | `gpu_stats_computer.py:150` | `(n_layers, n_feat_blocks, n_heads, 6)` |
| GPU stats (items) | `gpu_stats_computer.py:91` | `(n_layers, n_blocks, n_heads, 3)` |

---

## 🧮 FINAL FEATURE VECTOR DIMENSIONS

### Composition
```
per_feature_vector = [between_features_stats] + [between_items_stats] 
                   + [embeddings] + [gradients] + [mlp_activations]

Basic mode (enriched=False):
  Between-features: 12 layers × 6 heads × 6 stats + 3 cross-layer = 435 dims
  Between-items:    12 layers × 6 heads × 3 stats = 216 dims
  Embeddings:       2 × 192 (test) + 192 (train) = 576 dims
  Input gradients:  4 dims (mean, max, std, polarity)
  MLP activations:  12 layers × 5 stats = 60 dims
  
  TOTAL ≈ 1,300 dims/feature (with all categories)

Enriched mode (enriched=True):
  Between-features: 12 × 6 × 15 + 3 = 1,083 dims
  Between-items:    12 × 6 × 9 = 648 dims
  (rest same)
  
  TOTAL ≈ 2,400 dims/feature
```

---

## 🔍 HOW ATTENTION SHAPES FLOW

```
Raw input: (seq_len, batch, n_features=7)
    ↓ [pad, group by features_per_group=2]
Grouped: (batch, seq_len, n_feature_blocks=4, features_per_group=2)
    ↓ [flatten, encode]
Encoded: (batch, seq_len, n_feature_blocks=4, emsize=192)
    ↓ [concat with target y]
Combined: (batch, seq_len, n_blocks=5, emsize=192)
    ↓ [per-layer attention]

Between-features attn:
  Reshape to flat batch, compute attention
  Q, K: (batch, n_blocks=5, n_heads=6, d_k=32)
  Output: (batch, 5, 5, 6) before mean over batch
  
Between-items attn:
  Transpose sequence dimension
  Input: (batch, n_blocks=5, seq_len, emsize)
  Q, K: (batch*5, seq_len, n_heads=6, d_k=32)
  Output: (n_blocks=5, seq_len, seq_len, 6) for all items

Signal extraction:
  Extract and cache attention weights
  Shape: (batch, n_blocks, n_blocks, n_heads) for between-features
  Shape: (n_blocks, n_items, n_items, n_heads) for between-items

Signal processing:
  Mean over batch items, compute per-block stats
  Gather stats to per-feature level using block indices
  Final: (n_features, n_layers*n_heads*n_stats + cross_layer)
```

---

## ✨ ENRICHED VS BASIC MODE

### What Changes
```
Basic (enriched=False):
  - Between-features: 6 stats/head
  - Between-items: 3 stats/head (only if no train/test split)
  - Minimal memory overhead

Enriched (enriched=True):
  - Between-features: 15 stats/head (adds asymmetry, ranking, higher-order stats)
  - Between-items: 9 stats/head (adds train/test dynamics when applicable)
  - ~2.5x more feature dimensions
  - Better signal for interpretation model
```

### Enriched-Only Stats
```
Between-features:
  7. max_to_others
  8. std_to_others
  9. max_from_others
  10. std_from_others
  11. mean_asymmetry (out - in)
  12. max_abs_asymmetry
  13. target_out_rank (% blocks with less attention than target)
  14. target_in_rank
  15. contrast_to_target (to_target - mean_to_others)

Between-items:
  4. mean_train_to_test
  5. mean_test_to_train
  6. mean_self_train
  7. mean_self_test
  8. train_test_ratio
  9. concentration (max / row_sum)
```

