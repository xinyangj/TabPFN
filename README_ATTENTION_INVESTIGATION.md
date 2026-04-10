# TabPFN Attention & Feature Extraction Investigation

## 📚 Documentation Index

This investigation provides comprehensive documentation of how TabPFN's attention mechanism and feature extraction pipeline works. Four documents have been created:

### 1. 🎯 START HERE: `INVESTIGATION_SUMMARY.md`
**Best for:** Quick answers to your original questions
- Answers all 6 main questions you asked
- File locations and code references
- Dimension summary tables
- Quick-start usage guide
- 25 KB, easy to skim

### 2. 🔍 DEEP DIVE: `ATTENTION_AND_EXTRACTION_GUIDE.md`
**Best for:** Complete understanding of all mechanisms
- Full architectural explanations
- Detailed algorithm descriptions
- Tensor shapes at every step
- Enriched vs basic mode comparison
- GPU optimization details
- 39 KB, comprehensive reference

### 3. ⚡ QUICK LOOKUP: `QUICK_REFERENCE_ATTENTION.md`
**Best for:** Fast reference while coding
- Shape cheat sheets
- Dimension tables
- Key insights highlighted
- Code location index
- 4 KB, highly scannable

### 4. 💻 PRACTICAL EXAMPLES: `CODE_EXAMPLES_ATTENTION.md`
**Best for:** Understanding through working code
- 12 complete code examples
- How-to guides
- Memory optimization examples
- Complete end-to-end pipeline
- 12 KB, runnable snippets

---

## 🎯 Quick Answers (Full Answers in INVESTIGATION_SUMMARY.md)

### How does TabPFN encode tabular data?
- Features grouped into blocks of size `features_per_group` (default=2)
- Example: 7 features → 4 blocks (grouping by 2) + 1 target block = 5 total
- Encoded: `(batch, seq_len, n_feature_blocks, emsize=192)`

### What are the attention dimensions?
- Default: `emsize=192, nhead=6, nlayers=12`
- Per-head: `d_k=d_v=32`
- Shape: `(batch, n_blocks, n_blocks, n_heads)`

### Between-Features vs Between-Items Attention?

**Between-Features** (intra-sample):
- Attention across feature blocks within one sample
- Shape: `(batch, n_blocks, n_blocks, n_heads)` = `(batch, 5, 5, 6)`
- Purpose: Learn feature dependencies

**Between-Items** (inter-sample):
- Attention across samples for each feature block
- Shape: `(n_blocks, n_items, n_items, n_heads)` = `(5, 120, 120, 6)`
- Purpose: Learn sample relationships

### What stats are extracted?

**Between-Features (basic mode: 6 stats/head)**
1. Self-attention, 2. To-target, 3. From-target, 4. Mean-to-others, 5. Mean-from-others, 6. Entropy

**Between-Items (basic mode: 3 stats/head)**
1. Entropy, 2. Max, 3. Variance

**Enriched mode** adds 9 and 6 additional stats respectively

### Final feature vector dimensions?
- Basic mode: `(n_features, ~1,300)` all categories
- Enriched mode: `(n_features, ~2,400)` all categories

---

## 🗂️ Documentation Files

```
TabPFN/
├── README_ATTENTION_INVESTIGATION.md  ← You are here
├── INVESTIGATION_SUMMARY.md            ← Answer sheet for your questions
├── ATTENTION_AND_EXTRACTION_GUIDE.md   ← Technical deep dive (39 KB)
├── QUICK_REFERENCE_ATTENTION.md        ← Cheat sheet & tables (4 KB)
├── CODE_EXAMPLES_ATTENTION.md          ← Working code examples (12 KB)
│
└── src/tabpfn/
    ├── architectures/base/
    │   ├── config.py                   ← Model configuration
    │   ├── transformer.py              ← Feature grouping & forward pass
    │   ├── layer.py                    ← Between-features/items attention
    │   └── attention/full_attention.py ← Q,K,V computation & caching
    └── interpretation/extraction/
        ├── signal_extractor.py         ← Attention hook registration
        ├── signal_processor.py         ← Per-feature stats computation
        └── gpu_stats_computer.py       ← Memory-efficient GPU stats
```

---

## 🚀 Reading Guide by Use Case

### "I need a quick answer"
1. Read: `INVESTIGATION_SUMMARY.md` (15 min)

### "I need to understand the mechanism"
1. Read: `ATTENTION_AND_EXTRACTION_GUIDE.md` (30 min)
2. Reference: `QUICK_REFERENCE_ATTENTION.md` (5 min)

### "I need to write code using this"
1. Skim: `CODE_EXAMPLES_ATTENTION.md` (10 min)
2. Reference: `QUICK_REFERENCE_ATTENTION.md` (on demand)

### "I need complete understanding"
1. Read: `INVESTIGATION_SUMMARY.md` (15 min)
2. Read: `ATTENTION_AND_EXTRACTION_GUIDE.md` (30 min)
3. Work through: `CODE_EXAMPLES_ATTENTION.md` (30 min)

---

## 📊 Key Metrics at a Glance

| Metric | Value |
|--------|-------|
| Feature grouping default | 2 |
| Attention heads | 6 |
| Embedding dimension | 192 |
| Per-head dimension | 32 (= 192÷6) |
| Transformer layers | 12 |
| Between-features basic stats/head | 6 |
| Between-items basic stats/head | 3 |
| Basic mode per-feature dims | ~1,300 |
| Enriched mode per-feature dims | ~2,400 |
| Raw attention→GPU stats compression | ~1,000x |

---

## 🔗 Key Concepts Explained

### Features_per_group
- Controls how features are grouped into blocks
- Default: 2 (group consecutive pairs)
- Formula: `n_feature_blocks = ceil(n_features / features_per_group)`

### Block
- A group of `features_per_group` consecutive features
- Last block always contains target variable
- Example: feature [0,1], [2,3], [4,5], [6], [target]

### Between-Features Attention
- Attention mechanism operating on feature blocks
- Within single sample/item
- Learns feature dependencies

### Between-Items Attention
- Attention mechanism operating on samples
- Within single feature block
- Learns sample relationships (which training samples are similar)

### Signal Extraction
- Process of capturing internal representations (attention, embeddings, MLP outputs)
- Multiple gradient modes for speed/completeness trade-off

### Signal Processing
- Aggregating raw signals into fixed-size per-feature vectors
- Two modes: basic (fast) and enriched (detailed)

### GPU Stats Computer
- Memory-efficient path: compute stats on GPU, transfer only final statistics
- ~1 GB raw attention → ~20 KB stats

---

## 🎨 Architecture Visualization

```
Raw Tabular Data
    ↓
Group Features (features_per_group=2)
    ↓
Encode Each Feature Block (emsize=192)
    ↓
Add Target Block
    ↓
Transformer Encoder (12 layers)
    ├─ Between-features attention (feature block interactions)
    │   └─ Shape: (batch, n_blocks, n_blocks, n_heads)
    │
    ├─ Between-items attention (sample interactions)
    │   └─ Shape: (n_blocks, n_items, n_items, n_heads)
    │
    └─ MLP (feedforward)
        └─ Activations captured per layer
    ↓
Extract Signals (attention weights, embeddings, gradients, MLP outputs)
    ↓
Compute Per-Block Statistics (GPU-accelerated)
    ├─ Between-features: 6-15 stats/head
    ├─ Between-items: 3-9 stats/head
    └─ MLP: 5 stats/layer
    ↓
Map Blocks → Features (via block indices)
    ↓
Final Per-Feature Vectors (1,300-2,400 dims each)
    ↓
Use in Interpretation Model
```

---

## 💡 Important Notes

1. **Model Configuration Varies**: Default config has specific values, but pre-trained models might differ

2. **Memory Efficiency**: Use `GPUStatsComputer` for large datasets - it compresses attention tensors by ~1000x

3. **Gradient Modes**:
   - `True`: Full gradients, slowest
   - `"input_only"`: Input gradients only, medium
   - `False`: No gradients, fastest

4. **Enriched vs Basic**:
   - Basic: Lower dimension (1,300), faster
   - Enriched: Higher dimension (2,400), more information

5. **Block to Feature Mapping**: Via `bi_arr` array in signal processor - maps each feature to its block

---

## 📞 Document Navigation

**From INVESTIGATION_SUMMARY.md:**
- Jump to: `ATTENTION_AND_EXTRACTION_GUIDE.md` for full details
- Jump to: `CODE_EXAMPLES_ATTENTION.md` for working code

**From ATTENTION_AND_EXTRACTION_GUIDE.md:**
- Jump to: `QUICK_REFERENCE_ATTENTION.md` for quick lookups
- Jump to: `CODE_EXAMPLES_ATTENTION.md` for implementation

**From CODE_EXAMPLES_ATTENTION.md:**
- Jump to: `QUICK_REFERENCE_ATTENTION.md` for dimension reference
- Jump to: `ATTENTION_AND_EXTRACTION_GUIDE.md` for algorithm details

---

## ✅ Checklist: What You Now Understand

- [ ] How features are grouped into blocks
- [ ] What the target variable block is and why
- [ ] Between-features attention (shape, purpose, computation)
- [ ] Between-items attention (shape, purpose, computation)
- [ ] How attention weights are extracted via hooks
- [ ] Three gradient extraction modes and their trade-offs
- [ ] Per-feature statistics computation (basic & enriched)
- [ ] Block→feature mapping mechanism
- [ ] GPU stats optimization for memory efficiency
- [ ] Final per-feature vector dimensions and composition

---

## 🎓 Advanced Topics (For Later)

If you want to go deeper, the guide covers:
- Straight-through estimator for SDPA gradients
- Multiquery attention for test set
- Activation checkpointing for memory
- Feature positional embeddings (subspace mode)
- Cross-layer summary statistics computation
- Concentration and asymmetry metrics

---

## 📝 Citation Reference

When referencing this investigation:

**File Locations:**
- Main transformer: `src/tabpfn/architectures/base/`
- Attention module: `src/tabpfn/architectures/base/attention/full_attention.py`
- Signal extraction: `src/tabpfn/interpretation/extraction/`

**Key Classes:**
- `PerFeatureTransformer` (transformer.py)
- `PerFeatureEncoderLayer` (layer.py)
- `MultiHeadAttention` (full_attention.py)
- `SignalExtractor` (signal_extractor.py)
- `SignalProcessor` (signal_processor.py)
- `GPUStatsComputer` (gpu_stats_computer.py)

---

Generated: 2024
Investigation Type: Comprehensive architecture & pipeline analysis
Scope: TabPFN attention mechanisms and feature extraction for interpretation model input

