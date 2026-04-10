# TabPFN Attention & Feature Extraction - Investigation Summary

## 📋 Documents Created

This investigation has produced 4 comprehensive documents:

1. **`ATTENTION_AND_EXTRACTION_GUIDE.md`** (39 KB)
   - Complete architectural guide with all details
   - Tensor shapes at each step
   - Full algorithm descriptions for stats computation
   - Reference material for deep dives

2. **`QUICK_REFERENCE_ATTENTION.md`** (4 KB)
   - Quick lookup for key concepts
   - Shape cheat sheet
   - Dimension tables
   - Enriched vs basic mode comparison

3. **`CODE_EXAMPLES_ATTENTION.md`** (12 examples)
   - Practical code snippets
   - How-to guides for extraction
   - Memory optimization examples
   - Complete end-to-end pipeline examples

4. **`INVESTIGATION_SUMMARY.md`** (this file)
   - Summary of key findings
   - File references and locations
   - Quick answers to your questions

---

## ✅ ANSWERS TO YOUR QUESTIONS

### 1. TabPFN's Transformer Architecture

**Q: How does TabPFN encode tabular data into the transformer?**
- Input shape: `(seq_len, batch, n_features)`
- Features are grouped into blocks of size `features_per_group` (default=2)
- Encoded to: `(batch, seq_len, n_feature_blocks, emsize=192)`
- Location: `transformer.py:369-395`

**Q: What is `features_per_group=3` and how does it relate to blocks?**
- Default is 2, not 3 (from `config.py:30`)
- It determines block size: `n_feature_blocks = ceil(n_features / features_per_group)`
- Example: 7 features + features_per_group=2 → 4 feature blocks
- Each block groups consecutive features for joint attention

**Q: How is the target variable represented?**
- Target becomes an extra "feature block" appended at the end
- Always at index `n_blocks - 1`
- Formula: `n_blocks = n_feature_blocks + 1`
- Location: `transformer.py:514-515`

**Q: What are the attention dimensions?**
- Default config (actual trained model may differ):
  - `emsize = 192`
  - `nhead = 6` (not 3 or 18)
  - `nlayers = 12` (not 18)
  - `d_k = d_v = 192 // 6 = 32`
- Per layer: 6 attention heads, each with 32-dim Q,K,V
- Location: `config.py:28-34, 57`

**Q: How is attention structured?**
- Two separate attentions per layer:
  1. **Between-features**: attention across feature blocks (intra-sample)
  2. **Between-items**: attention across samples (inter-sample)
- Location: `layer.py:306-368`

---

### 2. Between-Features Attention

**Q: What does "between features" mean?**
- Attention across different **feature blocks** within a **single sample**
- Question: "How much should feature block i attend to feature block j?"
- Used to model inter-feature dependencies
- Example: "Does feature block 0 rely on information from feature block 3?"

**Q: What is its shape?**
- Shape: `(batch_items, n_blocks, n_blocks, n_heads)`
- Example with 7 features: `(batch, 5, 5, 6)`
  - 5 blocks = 4 feature blocks + 1 target
  - 6 heads
- Location: `signal_extractor.py:381-387`

**Q: How is it computed?**
```
Q, K, V = linear_projections(input)  # (batch, n_blocks, n_heads, d_k)
logits = einsum("...qhd,...khd->...qkh", Q, K)  # (batch, n_blocks, n_blocks, n_heads)
logits *= sqrt(1/d_k)
attention_weights = softmax(logits, dim=2)  # Softmax over keys
output = einsum("...qkh,...khd->...qhd", attention_weights, V)
```
- Location: `full_attention.py:752-756`

---

### 3. Between-Items Attention

**Q: What does "between items" mean?**
- Attention across different **samples/rows** for the **same feature block**
- Question: "How much should sample j attend to sample k for feature block i?"
- Used to model sample relationships (e.g., which training samples are similar)

**Q: What is its shape?**
- Shape: `(n_blocks, n_items, n_items, n_heads)`
- Example: `(5, 120, 120, 6)`
  - 5 blocks = 4 feature blocks + 1 target
  - 120 items = 100 train + 20 test
  - 6 heads
- Location: `signal_extractor.py:389-392`

**Q: How is it computed?**
- Same Q,K,V mechanism as between-features, but:
  - Sequence dimension is transposed to `n_blocks` from `n_items`
  - Input shape: `(batch, n_items, n_blocks, emsize)`
  - Transposed to: `(batch, n_blocks, n_items, emsize)` for attention
  - Result shape: `(batch, n_blocks, n_items, n_items, n_heads)`
- Location: `layer.py:315-368`

---

### 4. Signal Extraction (`signal_extractor.py`)

**Q: How are raw attention tensors captured?**
- Via internal caching within attention forward pass
- When enabled, attention weights are computed and stored in `_cached_attention_weights`
- Then retrieved via `get_attention_weights()`

**Q: What hooks are used?**
- **Attention hooks**: Via `enable_attention_weights_return(True)` on attention modules
  - No actual PyTorch hooks, just flag-based caching
- **MLP activation hooks**: Via `register_forward_hook()` on MLP modules
  - Captures output of each layer's MLP during forward pass
- Location: `signal_extractor.py:163-203`

**Q: What is `input_only` mode?**
- `extract_gradients="input_only"`
- Extracts only `d(loss)/d(input)` gradients (not attention weight gradients)
- Uses pure SDPA (no `retain_grad` on attention weights)
- Attention weight *values* still computed post-hoc from detached Q, K
- Faster than full gradient mode, loses 432/1591 dims
- Location: `signal_extractor.py:119-129, 316-363`

---

### 5. Signal Processing (`signal_processor.py`)

**Q: What stats are computed per feature in `_process_between_features_attention()`?**

Basic mode (6 stats/head):
1. Self-attention (diagonal element)
2. Attention to target block
3. Attention from target block
4. Mean attention to other feature blocks
5. Mean attention from other feature blocks
6. Entropy

Enriched mode (9 additional):
7-10. Max/std to/from others
11-12. Asymmetry metrics
13-14. Target-relative ranking
15. Contrast to target

Location: `signal_processor.py:385-450`

**Q: What stats are computed per feature in `_process_between_items_attention()`?**

Basic mode (3 stats/head):
1. Entropy (row-wise)
2. Max attention
3. Variance

Enriched mode (6 additional, if train/test split):
4-7. Train→test, test→train, self-train, self-test means
8. Train/test attention ratio
9. Concentration metric

Location: `signal_processor.py:520-551`

**Q: How does block→feature mapping work?**
```python
n_feature_blocks = n_blocks - 1
bi_arr = [min(i * n_feature_blocks // n_features, n_feature_blocks - 1)
          for i in range(n_features)]
gathered = block_stats[:, bi_arr, :, :]  # Gather to per-feature level
```
Location: `signal_processor.py:372-373`

---

### 6. GPU Stats Computation (`gpu_stats_computer.py`)

**Q: How are the same stats computed on GPU?**
- Same formulas but using torch operations for speed
- Computed before CPU transfer (~1 GB raw attention → ~20 KB stats)
- All computation done on GPU, results transferred to CPU as numpy arrays

**Q: What's the shape flow?**
```
Raw attention:     (batch_items, n_blocks, n_blocks, n_heads) per layer
                   (n_blocks, n_items, n_items, n_heads) per layer

GPU stats:         (n_layers, n_feature_blocks, n_heads, 6)
                   (n_layers, n_blocks, n_heads, 3)

CPU gathering:     Block→feature mapping via bi_arr indexing

Per-feature vector: (n_features, n_layers*n_heads*n_stats + cross_layer)
                    = (n_features, 432) basic or (n_features, 1083) enriched
```
Location: `gpu_stats_computer.py:91-283`

---

## 🗂️ FILE STRUCTURE OVERVIEW

```
src/tabpfn/
├── architectures/base/
│   ├── transformer.py          ← Feature grouping, forward pass
│   ├── layer.py                ← Between-features & between-items attention layers
│   ├── attention/
│   │   └── full_attention.py   ← Q,K,V computation, SDPA, attention weight caching
│   ├── config.py               ← Model configuration (emsize, nhead, nlayers, features_per_group)
│   └── encoder.py              ← Linear encoder
├── interpretation/extraction/
│   ├── signal_extractor.py     ← Hook registration, signal collection
│   ├── signal_processor.py     ← Per-feature statistics computation
│   └── gpu_stats_computer.py   ← GPU-accelerated stats (memory efficient)
```

---

## 🧮 DIMENSION SUMMARY TABLE

| Component | Shape | Size | Notes |
|-----------|-------|------|-------|
| Raw attention (between-features) | (batch, n_blocks, n_blocks, n_heads) | (1, 5, 5, 6) | Per layer |
| Raw attention (between-items) | (n_blocks, n_items, n_items, n_heads) | (5, 120, 120, 6) | Per layer |
| GPU block stats (between-features) | (n_layers, n_feature_blocks, n_heads, 6) | (12, 4, 6, 6) | Memory efficient |
| GPU block stats (between-items) | (n_layers, n_blocks, n_heads, 3) | (12, 5, 6, 3) | Memory efficient |
| Per-feature vector (basic) | (n_features, dims) | (7, 1300) | All signal categories |
| Per-feature vector (enriched) | (n_features, dims) | (7, 2400) | Enhanced statistics |

---

## 🔑 KEY CODE REFERENCES

| Concept | File | Lines | Key Variables |
|---------|------|-------|----------------|
| Feature grouping | transformer.py | 369-395 | `features_per_group`, `n_feature_blocks` |
| Between-features attention | layer.py | 306-313 | `self_attn_between_features` |
| Between-items attention | layer.py | 315-368 | `self_attn_between_items` |
| Attention computation | full_attention.py | 752-772 | `logits`, `attention_weights` |
| Attention caching | full_attention.py | 565-571 | `_cached_attention_weights`, `retain_grad()` |
| Signal extraction | signal_extractor.py | 163-203 | `enable_attention_weights_return()` |
| Between-feat stats | signal_processor.py | 385-450 | Block stats computation |
| Between-items stats | signal_processor.py | 520-551 | Item attention stats |
| Block→feature map | signal_processor.py | 372-373 | `bi_arr` array |
| GPU stats (fast) | gpu_stats_computer.py | 91-283 | Per-block aggregation |
| Config defaults | config.py | 28-57 | `emsize`, `nhead`, `nlayers` |

---

## 🚀 QUICK START: How to Use

### 1. Extract Signals
```python
from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor

extractor = SignalExtractor(extract_gradients=True)  # Full gradients
signals = extractor.extract(model, X_train, y_train, X_test)
```

### 2. Process into Feature Vectors
```python
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor

processor = SignalProcessor(enriched=True)
feature_vectors = processor.process(signals)  # (n_features, ~2400)
```

### 3. Or Use Fast GPU Path
```python
from tabpfn.interpretation.extraction.gpu_stats_computer import GPUStatsComputer

gpu_computer = GPUStatsComputer(enriched=True)
gpu_stats = gpu_computer.compute(signals)
feature_vectors = processor.process_from_stats(gpu_stats, n_features=7)
```

### 4. Use Feature Vectors
```python
# These become input to interpretation model explaining feature importance
interpretation_model.fit(feature_vectors, feature_importance_labels)
```

---

## 📊 ENRICHED MODE COMPARISON

| Aspect | Basic | Enriched |
|--------|-------|----------|
| Between-features stats/head | 6 | 15 |
| Between-items stats/head | 3 | 9 |
| Per-feature dimensions | ~1,300 | ~2,400 |
| Memory overhead | Low | Medium |
| Information content | Baseline | Enhanced analysis |
| Recommended for | Quick analysis | Detailed interpretation |

---

## 🔗 Related Components (Not Covered in Depth)

- **Y-Encoder** (`encoders.py`): Encodes target variable into embedding space
- **Interpretation Model** (`interpretation/model/`): Neural network that predicts feature importance from signals
- **Feature Positional Embeddings** (subspace mode): Adds feature identity information
- **NaN Handling** (encoders): Preprocessing for missing values
- **Batch Normalization** (normalize_feature_groups): Feature normalization

---

## 📝 Notes & Caveats

1. **Configuration May Vary**: The default config has `nhead=6, nlayers=12`, but trained models may use different values.

2. **Straight-Through SDPA**: When `extract_gradients=True`, the SDPA forward uses the fused kernel for speed, but backward goes through the manual `softmax(QK^T)@V` for exact gradients.

3. **Block to Feature Mapping**: Not a one-to-one mapping. Multiple features may map to the same block depending on their positions.

4. **Memory Efficiency**: Use `GPUStatsComputer` for large datasets (~1000x compression of raw attention tensors).

5. **Train/Test Split**: Between-items enriched stats are only computed if both train and test samples exist and are separated.

---

## 📞 For Further Questions

Refer to the other documentation files:
- **Deep technical details**: `ATTENTION_AND_EXTRACTION_GUIDE.md`
- **Quick lookups**: `QUICK_REFERENCE_ATTENTION.md`
- **Code examples**: `CODE_EXAMPLES_ATTENTION.md`

