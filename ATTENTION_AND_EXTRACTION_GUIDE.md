# TabPFN Attention Architecture and Feature Extraction Guide

## 1. TABPFN'S TRANSFORMER ARCHITECTURE

### 1.1 Input Encoding into Blocks

**Data Flow:**
- Raw input: `X` of shape `(seq_len, batch, n_features)`
- Features are **grouped into blocks** of size `features_per_group` (default=2, configurable)
- Formula: `n_feature_blocks = ceil(n_features / features_per_group)`
- **Target variable** is added as an extra block at the end: `n_blocks = n_feature_blocks + 1`

**Example with n_features=7, features_per_group=2:**
- Block 0: features 0-1
- Block 1: features 2-3
- Block 2: features 4-5
- Block 3: features 6
- Block 4: target (y) ← last block is always target

**Encoding Process** (from `transformer.py:390-395`):
```python
# Input shape: (seq_len, batch, n_features)
# After padding to multiple of features_per_group:
x = rearrange(x, "s b (f n) -> b s f n", n=features_per_group)
# Result: (batch, seq_len, n_feature_blocks, features_per_group)

# Then reshaped to:
x = rearrange(x, "b s f n -> s (b f) n")
# (seq_len, batch*n_feature_blocks, features_per_group)
```

After linear encoder: `(seq_len, batch*n_feature_blocks, emsize=192)`

Then reshaped back to: `(batch, seq_len, n_feature_blocks, emsize)`

### 1.2 Attention Dimensions

**Model Configuration (defaults):**
- `emsize = 192` (embedding dimension per head)
- `nhead = 6` (attention heads, though you mentioned 3 heads and 18 layers)
- `nlayers = 12` (transformer layers, though you mentioned 18 layers)
- `d_k = d_v = emsize // nhead = 192 // 6 = 32` (per-head dimensions)

**Full per-head calculation:**
- Q, K, V dimensions: `(batch, seq_len, nhead, d_k) = (batch, seq_len, 6, 32)`
- Attention logits: `Q @ K^T` → `(batch, seq_len, seq_len, nhead)` after softmax
- Final output: `(batch, seq_len, nhead, d_v)` aggregated → `(batch, seq_len, emsize)`

---

## 2. BETWEEN-FEATURES ATTENTION

### 2.1 Definition and Purpose

**"Between-features"** means attention **across different feature blocks** within a single sample/item.

**Conceptual Flow:**
- Input: All feature blocks + target block for one sample
- Attention: How much each feature block attends to each other feature block
- Purpose: Learn inter-feature dependencies (which features are related)

### 2.2 Attention Matrix Shape and Computation

**Shape:** `(batch_items, n_blocks, n_blocks, n_heads)`
- `batch_items = batch_size` (or `batch_size * seq_len` if flattened)
- `n_blocks = n_feature_blocks + 1` (last block is target)
- `n_heads = 6`

**Example:** If `n_features=7, features_per_group=2`:
- `n_blocks = 5` (4 feature blocks + 1 target)
- Shape = `(batch, 5, 5, 6)`

**Computation** (from `attention.py:752-756` and `752-772`):
```python
# Q, K, V are computed from input: (batch, n_blocks, nhead, d_k)
# Logits: element-wise Q·K^T
logits = einsum("b q h d, b k h d -> b q k h", Q, K)  # (batch, n_blocks, n_blocks, nhead)

# Scale by sqrt(1/d_k)
logits = logits * scale  # scale = 1/sqrt(32) ≈ 0.176

# Softmax over key dimension (dim=2)
attention_weights = softmax(logits, dim=2)  # (batch, n_blocks, n_blocks, nhead)

# Apply to values
output = einsum("b q k h, b k h d -> b q h d", attention_weights, V)
```

### 2.3 Between-Features Attention Extraction in Signal Extractor

**Location:** `signal_extractor.py:381-387`

```python
for i, layer in enumerate(encoder_layers):
    if layer.self_attn_between_features is not None:
        attn = layer.self_attn_between_features.get_attention_weights()
        # attn shape: (batch, n_blocks, n_blocks, n_heads)
        between_features_attention[f"layer_{i}"] = attn.detach().clone()
```

**Hooks:** Via `enable_attention_weights_return(True)` on each layer's attention module
- When enabled, attention weights are computed and cached
- With `retain_grad=True`, gradients are also computed

---

## 3. BETWEEN-ITEMS ATTENTION

### 3.1 Definition and Purpose

**"Between-items"** means attention **across different samples/rows** for the same feature block.

**Conceptual Flow:**
- Input: All samples (train + test) for one feature block
- Attention: How much each sample attends to each other sample
- Purpose: Learn sample relationships (e.g., which training samples are similar to test samples)

### 3.2 Attention Matrix Shape and Computation

**Shape:** `(n_blocks, n_items, n_items, n_heads)`
- `n_blocks = n_feature_blocks + 1` (one per feature block + target)
- `n_items = n_train + n_test` (all samples)
- `n_heads = 6`

**Example:** If `n_train=20, n_test=5, n_features=7, features_per_group=2`:
- `n_blocks = 5` (4 feature blocks + 1 target)
- `n_items = 25` (20 train + 5 test)
- Shape = `(5, 25, 25, 6)`

**Transposition for Attention:**
```python
# Input x shape: (batch, n_items, n_blocks, emsize)
# For between-items attention, transpose to sequence dimension:
x_for_attn = x.transpose(1, 2)  # (batch, n_blocks, n_items, emsize)

# Now attention operates:
# Q, K, V: (batch, n_blocks, n_items, nhead, d_k)
# Attention: (batch, n_blocks, n_items, n_items, nhead)
```

From `layer.py:360-361`:
```python
return self.self_attn_between_items(
    x.transpose(1, 2),  # sequence dimension = n_blocks, n_items becomes features
    ...
).transpose(1, 2)
```

### 3.3 Between-Items Attention Extraction

**Location:** `signal_extractor.py:389-392`

```python
attn_items = layer.self_attn_between_items.get_attention_weights()
between_items_attention[f"layer_{i}"] = attn_items.detach().clone()
# Shape: (n_blocks, n_items, n_items, n_heads)
```

---

## 4. SIGNAL EXTRACTION PIPELINE

### 4.1 SignalExtractor Overview

**Location:** `signal_extractor.py`

**Purpose:** Extract internal representations from TabPFN during inference

**Three Gradient Modes:**
1. **`extract_gradients=True`** (default):
   - Full gradient computation with straight-through SDPA
   - Attention weight gradients are computed via shadow softmax(QK^T)@V
   - All 436 gradient dimensions included
   - **Straight-through estimator:** SDPA forward values, backward through manual attention
   
2. **`extract_gradients="input_only"`**:
   - Only `d(loss)/d(input)` gradients extracted
   - Pure SDPA (no `retain_grad` on attention weights)
   - Attention weight *values* still computed post-hoc from detached Q, K
   - Drops 432/1591 dims (attention_gradient stats)
   - Faster than full gradient mode
   
3. **`extract_gradients=False`**:
   - No backward pass at all
   - Fastest but loses all 436 gradient dimensions
   - Still extracts attention weights (values only)

### 4.2 Hook Registration

**MLP Activation Hooks** (lines 187-203):
```python
def hook_fn(module, input, output, layer_idx=i):
    storage[layer_idx] = output.detach()  # Store MLP output

hooks.append(layer.mlp.register_forward_hook(hook_fn))
```

**Attention Weight Hooks** (via enable/disable):
```python
for layer in layers:
    layer.self_attn_between_features.enable_attention_weights_return(True)
    layer.self_attn_between_items.enable_attention_weights_return(True)
    if grad:
        layer.self_attn_between_features.enable_attention_grad_retention(True)
        layer.self_attn_between_items.enable_attention_grad_retention(True)
```

### 4.3 Signal Collection

**Extracted Signals Dictionary:**
```python
{
    "between_features_attention": {
        "layer_0": torch.Tensor,  # (batch_items, n_blocks, n_blocks, n_heads)
        "layer_1": torch.Tensor,
        ...
    },
    "between_items_attention": {
        "layer_0": torch.Tensor,  # (n_blocks, n_items, n_items, n_heads)
        "layer_1": torch.Tensor,
        ...
    },
    "train_embeddings": torch.Tensor,  # (n_train, emsize)
    "test_embeddings": torch.Tensor,   # (n_test, emsize)
    "input_gradients": torch.Tensor,   # (n_samples, n_features)
    "attention_gradients": {
        "layer_0": torch.Tensor or None,  # (batch_items, n_blocks, n_blocks, n_heads)
        ...
    },
    "mlp_activations": {
        0: torch.Tensor,  # (batch, batch_items, n_blocks, emsize)
        1: torch.Tensor,
        ...
    },
    "n_features": int,
    "n_train": int,
    "n_test": int,
}
```

---

## 5. SIGNAL PROCESSING - STATS COMPUTATION

### 5.1 Between-Features Attention Processing

**Location:** `signal_processor.py:338-478`

**Input Shape:** `(batch_items, n_blocks, n_blocks, n_heads)` per layer

**Output Shape Per Feature:** `(n_layers * n_heads * n_stats) + n_cross_layer_stats`

**Step 1: Mean over batch_items**
```python
mean_attns_np = [(attn.mean(dim=0)).numpy() for attn in layer_attns]
# Result: list of (n_blocks, n_blocks, n_heads) arrays
```

**Step 2: Entropy Computation**
```python
# For each layer, head:
a_clamped = clip(attention_weights, 1e-10, inf)
entropy = -(a_clamped * log(a_clamped)).sum(axis=1)  # (n_blocks, n_heads)
```

**Step 3: Per-Feature Statistics** (6 basic, 15 if enriched)

For each feature block `i`:

**Basic Stats (all modes):**
1. **self-attention:** `ah[i, i, h]` - diagonal element
2. **to_target:** `ah[i, target_block, h]` - attention to target block
3. **from_target:** `ah[target_block, i, h]` - attention from target block
4. **mean_to_others:** `(sum(ah[i, 0:n_fb]) - ah[i,i]) / (n_fb-1)` - mean attention to feature blocks (excluding self)
5. **mean_from_others:** `(sum(ah[0:n_fb, i]) - ah[i,i]) / (n_fb-1)` - mean attention from feature blocks (excluding self)
6. **entropy:** Per-feature entropy

**Enriched Stats (6-15):**
7. **max_to_others:** Max attention to other feature blocks
8. **std_to_others:** Std of attention to other feature blocks
9. **max_from_others:** Max attention from other feature blocks
10. **std_from_others:** Std of attention from other feature blocks
11. **mean_asymmetry:** Mean difference between outgoing/incoming attention
12. **max_abs_asymmetry:** Max absolute asymmetry
13. **target_out_rank:** Fraction of blocks with attention ≤ to_target
14. **target_in_rank:** Fraction of blocks with attention ≤ from_target
15. **contrast_to_target:** to_target - mean_to_others

**Step 4: Block→Feature Mapping**

```python
n_feature_blocks = n_blocks - 1  # Exclude target
bi_arr = np.array([min(i * n_feature_blocks // n_features, n_feature_blocks - 1)
                   for i in range(n_features)])
# Maps each feature to its corresponding block

# Gather: (n_layers, n_feature_blocks, n_heads, n_stats) 
#      → (n_layers, n_features, n_heads, n_stats)
gathered = block_stats[:, bi_arr, :, :]
```

**Step 5: Cross-Layer Summary Stats**

```python
to_target_per_layer = [a[:n_fb, target, :].mean(axis=-1) for a in mean_attns_np]
# (n_layers, n_fb) → (n_layers, n_features) after gathering

cross_layer[:, 0] = to_target_feat.mean(axis=0)  # mean across layers
cross_layer[:, 1] = to_target_feat.std(axis=0)   # std across layers
cross_layer[:, 2] = correlation(layer_idx, to_target)  # trend slope
```

**Final Shape Per Feature:**
```
n_layers * n_heads * n_stats_per_head + 3 cross_layer
= 12 * 6 * 6 + 3 = 435 dimensions (basic mode)
= 12 * 6 * 15 + 3 = 1083 dimensions (enriched mode)
```

### 5.2 Between-Items Attention Processing

**Location:** `signal_processor.py:480-555`

**Input Shape:** `(n_blocks, n_items, n_items, n_heads)` per layer

**Output Shape Per Feature:** `(n_layers * n_heads * n_stats)`

**Basic Stats (3: always computed):**
1. **entropy:** Row-wise entropy, mean over items `-(p*ln(p)).sum(dim=2).mean(dim=1)`
2. **max:** Max attention value across all item pairs
3. **variance:** Variance of flattened item×item attention matrix

**Enriched Stats (3-9, if has train/test split):**

If `n_train > 0` and `n_train < n_items`:
```python
t2t = ah[:, :n_train, n_train:]    # train→test attention
st = ah[:, :n_train, :n_train]     # self-train attention
t2s = ah[:, n_train:, :n_train]    # test→train attention
ss = ah[:, n_train:, n_train:]     # self-test attention

4. mean_train_to_test = t2t.mean(dim=(1,2))
5. mean_test_to_train = t2s.mean(dim=(1,2))
6. mean_self_train = st.mean(dim=(1,2))
7. mean_self_test = ss.mean(dim=(1,2))
8. train_test_ratio = mean_t2t / (mean_st + 1e-10)
9. concentration = max_val / (row_sum + 1e-10)  # how concentrated is attention
```

**Step 1: Block→Feature Mapping**
```python
bi_arr = np.array([min(i * n_blocks_attn // max(n_features, 1), n_blocks_attn - 1)
                   for i in range(n_features)])
gathered = block_stats[:, bi_arr, :, :]  # (n_layers, n_features, n_heads, n_stats)
```

**Final Shape Per Feature:**
```
n_layers * n_heads * n_stats
= 12 * 6 * 3 = 216 dimensions (basic mode)
= 12 * 6 * 9 = 648 dimensions (enriched mode)
```

---

## 6. GPU STATS COMPUTATION (Fast Path)

**Location:** `gpu_stats_computer.py`

**Purpose:** Compute all per-block statistics on GPU before transferring to CPU (~1 GB raw attention → ~20 KB stats)

### 6.1 Between-Items Attention Stats

**GPU Computation** (lines 91-128):
```python
# Input: (n_blocks, n_items, n_items, n_heads) tensors on GPU

for each layer:
    # Entropy: per-row sum of -p*ln(p)
    row_ent = torch.special.entr(at).sum(dim=2)  # (n_blocks, n_items, n_heads)
    entropy_per_block = row_ent.mean(dim=1)  # (n_blocks, n_heads)
    
    # Max and variance on flattened items
    af = at.reshape(n_blocks, -1, n_heads)  # (n_blocks, n_items*n_items, n_heads)
    max_vals = af.max(dim=1).values
    vars = af.var(dim=1)

# Output: (n_layers, n_blocks, n_heads, 3) numpy array
```

### 6.2 Between-Features Attention Stats

**GPU Computation** (lines 130-199):
```python
# Input: (batch_items, n_blocks, n_blocks, n_heads) tensors on GPU

for each layer:
    mean_at = at.mean(dim=0)  # (n_blocks, n_blocks, n_heads)
    
    for each head:
        ah = mean_at[:, :, h]  # (n_blocks, n_blocks)
        feat_ah = ah[:n_feature_blocks, :]
        
        # 1. self-attention: diagonal
        stats[:, 0] = diag(ah)[:n_feature_blocks]
        
        # 2. to_target, 3. from_target
        stats[:, 1] = feat_ah[:, target_block]
        stats[:, 2] = ah[target_block, :n_feature_blocks]
        
        # 4-5. mean to/from others
        row_sums = feat_ah.sum(dim=1)
        diag_vals = diag(feat_ah)
        stats[:, 3] = (row_sums - diag_vals) / max(n_fb-1, 1)
        col_sums = feat_ah.sum(dim=0)
        stats[:, 4] = (col_sums - diag_vals) / max(n_fb-1, 1)
        
        # 6. entropy
        a_clamped = clamp(mean_at[:n_fb, :, h], min=1e-10)
        stats[:, 5] = -(a_clamped * log(a_clamped)).sum(dim=1)

# Output: (n_layers, n_feature_blocks, n_heads, 6) numpy array
# Also: features_to_target_per_layer (n_layers, n_feature_blocks) for cross-layer
```

### 6.3 Shape Flow from Raw Attention to Final Vectors

**Raw attention tensors:**
```
Between-features: (batch_items, n_blocks, n_blocks, n_heads)
Between-items: (n_blocks, n_items, n_items, n_heads)
```

**GPU stats (per-block level):**
```
Between-features: (n_layers, n_feature_blocks, n_heads, 6)
Between-items: (n_layers, n_blocks, n_heads, 3)
```

**CPU gathering (block→feature mapping):**
```
bi_arr = [min(i * n_feature_blocks // n_features, n_feature_blocks - 1) 
          for i in range(n_features)]
gathered = stats[:, bi_arr, :, :]  # (n_layers, n_features, n_heads, stats)
```

**Per-feature vectors:**
```
Reshape: (n_features, n_layers * n_heads * stats)
Example: (n_features, 12 * 6 * 6) = (n_features, 432)
```

---

## 7. DETAILED SHAPES AT EACH STEP

### Forward Pass Shape Transformations

**Input:**
```
X: (seq_len, batch, n_features=7)
y: (seq_len, batch) or (n_train, batch)
```

**After padding to features_per_group=2:**
```
X: (seq_len, batch, 8)  # padded from 7→8
```

**After grouping (transformer.py:391-395):**
```
X: (batch, seq_len, 4, 2)  # (batch, seq_len, n_feature_blocks, features_per_group)
```

**After flattening batch×blocks, encoding:**
```
X: (seq_len, batch*4, 2)  # (seq_len, batch*n_feature_blocks, features_per_group)
embedded_x: (seq_len, batch*4, emsize=192)
```

**After reshaping back:**
```
embedded_x: (batch, seq_len, 4, emsize=192)  # (batch, seq_len, n_feature_blocks, emsize)
embedded_y: (batch, seq_len, 1, emsize=192)  # target added
```

**After concatenating x + y:**
```
embedded_input: (batch, seq_len, 5, emsize=192)  # (batch, seq_len, n_blocks, emsize)
                                                   # n_blocks = n_feature_blocks + 1
```

**In between-features attention:**
```
Input to attn: (batch, seq_len, 5, emsize=192)
Q, K, V computed: (batch, seq_len*5, nhead=6, d_k=32)
  (flattened via _rearrange_inputs_to_flat_batch)
Attention logits: (batch, 5, 5, 6)  # (batch, n_blocks, n_blocks, n_heads)
Attention weights: (batch, 5, 5, 6)
Output: (batch, seq_len, 5, emsize=192)
```

**In between-items attention:**
```
Input to attn: (batch, seq_len, 5, emsize=192)
After transpose(1,2): (batch, 5, seq_len, emsize=192)
Q, K, V computed: (batch*5, seq_len, nhead=6, d_k=32)
Attention logits: (batch*5, seq_len, seq_len, 6)
Reshaped to (batch, 5, seq_len, seq_len, 6)
After softmax and aggregation: (batch, 5, seq_len, emsize)
After transpose(1,2) back: (batch, seq_len, 5, emsize)
```

### Attention Extraction Shapes

**In signal_extractor.py:**
```
between_features_attention["layer_0"]:
  - Captured from layer.self_attn_between_features.get_attention_weights()
  - Shape: (batch_size, n_blocks, n_blocks, n_heads)
  - Example: (1, 5, 5, 6)

between_items_attention["layer_0"]:
  - Captured from layer.self_attn_between_items.get_attention_weights()
  - Shape: (n_blocks, n_items, n_items, n_heads)
  - Example: (5, 25, 25, 6)  # if n_items=25
```

---

## 8. ENRICHED vs BASIC MODE

### SignalProcessor Comparison

**Basic Mode** (`enriched=False`):
- Between-features: 6 stats/head
- Between-items: 3 stats/head
- Cross-layer: 3 summary stats
- **Total per-feature dim:** ~432 + 216 + embeddings + gradients

**Enriched Mode** (`enriched=True`):
- Between-features: 15 stats/head (additional 9: max/std to/from others, asymmetry, target ranking)
- Between-items: 9 stats/head (additional 6: train/test split statistics)
- Cross-layer: 3 summary stats
- **Total per-feature dim:** ~1083 + 648 + embeddings + gradients

### Which Stats Differ

**Between-features enriched additions:**
```python
7-10:   max/std to others, max/std from others (requires masking for each block)
11-12:  mean/max asymmetry (difference between outgoing/incoming)
13-14:  target-relative ranking (percentile within all blocks)
15:     contrast to target (to_target - mean_to_others)
```

**Between-items enriched additions** (only if has train/test split):
```python
4-7:    train→test, test→train, self-train, self-test means
8:      train/test attention ratio
9:      attention concentration metric
```

---

## 9. KEY CODE REFERENCES

### Tensor Shape Tracking

| Component | Location | Shape | Notes |
|-----------|----------|-------|-------|
| Between-features attn | `signal_extractor.py:382` | `(batch, n_blocks, n_blocks, n_heads)` | Per layer |
| Between-items attn | `signal_extractor.py:390` | `(n_blocks, n_items, n_items, n_heads)` | Per layer |
| MLP activations | `signal_extractor.py:193-202` | `(batch, batch_items, n_blocks, emsize)` | Per layer |
| Block stats (feat) | `gpu_stats_computer.py:156-159` | `(n_layers, n_feature_blocks, n_heads, 6)` | GPU computed |
| Block stats (items) | `gpu_stats_computer.py:115` | `(n_layers, n_blocks, n_heads, 3)` | GPU computed |
| Feature vector | `signal_processor.py:455-458` | `(n_features, n_layers*n_heads*n_stats+cross)` | Final output |

### Critical Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `enable_attention_weights_return()` | `full_attention.py:115` | Enable attention caching |
| `get_attention_weights()` | `full_attention.py:146` | Retrieve cached weights |
| `compute_attention_heads()` | `full_attention.py:665` | Compute logits, softmax, apply to values |
| `_process_between_features_attention()` | `signal_processor.py:338` | Compute all per-feature statistics |
| `_process_between_items_attention()` | `signal_processor.py:480` | Compute per-feature train/test stats |
| `_compute_features_attention_stats()` | `gpu_stats_computer.py:130` | GPU-accelerated stats computation |

