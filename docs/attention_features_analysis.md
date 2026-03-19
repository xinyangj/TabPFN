# Deep Dive: TabPFN Attention Features for Interpretation

## 1. How TabPFN Encodes Tabular Data

### Feature Grouping into Blocks

TabPFN does NOT process each tabular feature independently. It groups every `features_per_group` (=3 for v2.5) raw features into a single **feature block**, then treats each block as a "token" in the transformer.

```
Raw features:  [f0, f1, f2, f3, f4, f5, f6, f7]  (8 features)
                \_________/  \_________/  \____/
Padding:       [f0,f1,f2]   [f3,f4,f5]   [f6,f7,0]  (pad to multiple of 3)
                 block_0       block_1      block_2
Target:                                               [y]
                                                      block_3

n_feature_blocks = ceil(8/3) = 3
n_blocks = 3 + 1 = 4  (feature blocks + target block)
```

Each block is projected to `emsize=192` via a learned encoder. The target block encodes the label `y` (with NaN for test items). After encoding:

**State shape: `(batch_size=1, n_items, n_blocks=4, emsize=192)`**

where `n_items = n_train + n_test` (the rows/samples).

### Two Types of Attention

The transformer has **two separate attention operations per layer**, applied to different axes of the 4D state:

#### A. Between-Features Attention
- **Axis**: Operates across **feature blocks** within each item (sample)
- **Input**: `state` with shape `(batch_size, n_items, n_blocks, emsize)` — attention over dim 2 (n_blocks)
- **The attention module sees**: Each (batch, item) pair independently; Q,K,V come from blocks within that item
- **Attention weight shape**: `(batch_items, n_blocks, n_blocks, n_heads)` where `batch_items = batch_size × n_items`
- **What it captures**: "How much should feature block 0 attend to the target block?", "How much should feature block 1 attend to feature block 2?"
- **This is the most directly relevant attention for feature importance** — it represents how the model routes information between features and the target within each sample

```
Between-features attention matrix (per item):
                 block_0  block_1  block_2  target
    block_0    [ 0.30    0.15     0.25     0.30  ]  ← f0,f1,f2 look at...
    block_1    [ 0.10    0.40     0.10     0.40  ]  ← f3,f4,f5 look at...
    block_2    [ 0.20    0.10     0.35     0.35  ]  ← f6,f7 look at...
    target     [ 0.25    0.30     0.15     0.30  ]  ← target looks at...
```

#### B. Between-Items Attention
- **Axis**: Operates across **items** (samples/rows) within each feature block
- **Input**: `state.transpose(1,2)` → shape `(batch, n_blocks, n_items, emsize)` — attention over dim 2 (n_items)
- **The attention module sees**: Each (batch, block) pair independently; Q,K,V come from items within that block
- **Attention weight shape**: `(n_blocks, n_items, n_items, n_heads)` where the first dim is n_blocks (batch × n_blocks for batch>1)
- **What it captures**: "How much should test_sample_0 attend to train_sample_5 for feature block 1?", enabling ICL (in-context learning)

```
Between-items attention (for one feature block):
                 train_0  train_1  ...  test_0   test_1
    train_0    [ 0.20    0.30     ...   0.05     0.05  ]
    train_1    [ 0.35    0.15     ...   0.10     0.08  ]
    ...
    test_0     [ 0.30    0.25     ...   0.05     0.03  ]  ← ICL: test looks at train
    test_1     [ 0.28    0.22     ...   0.04     0.06  ]
```

### The Fundamental Resolution Problem: Block Quantization

Because `features_per_group=3`, features within the same block are **indistinguishable** at the attention level:

```
8 features → 3 blocks:
  f0, f1, f2 → block_0    (all three get IDENTICAL attention stats)
  f3, f4, f5 → block_1    (all three get IDENTICAL attention stats)
  f6, f7     → block_2    (both get IDENTICAL attention stats)
```

The `bi_arr` mapping: `bi_arr[i] = min(i * n_feature_blocks // n_features, n_feature_blocks - 1)` maps features to blocks, then `gathered = block_stats[:, bi_arr, :, :]` copies the same block stats to all features in that block.

**This is not a bug** — it's an inherent limitation of how TabPFN processes features. The attention mechanism literally cannot distinguish f0 from f1 because they're packed together in the same embedding. The model learns to process them jointly.

**Impact**: For 8 features → 3 blocks, at most 3 unique attention patterns exist. Features in the same block get identical feature vectors for attention-based categories (between-features attn, between-items attn, MLP activations, attention gradients).

Only **input gradients** (d(loss)/d(x)) are truly per-feature, since they differentiate w.r.t. each raw input before grouping.

## 2. What Statistics Are Extracted Per Category

### 2.1 Between-Features Attention — THE CORE SIGNAL

**Raw shape**: `(batch_items, n_blocks, n_blocks, n_heads)` per layer → 18 layers

#### Basic mode (enriched=False): 6 stats per layer×head → **327 dims**

| # | Stat | Formula | Interpretation |
|---|------|---------|----------------|
| 1 | self_attention | `diag(ah)[bi]` | How much this block attends to itself |
| 2 | to_target | `ah[bi, target]` | **Key signal**: How much this feature block pays attention to the target |
| 3 | from_target | `ah[target, bi]` | How much the target pays attention to this block |
| 4 | mean_to_others | `(row_sum - self) / (n_fb-1)` | Avg attention to other feature blocks |
| 5 | mean_from_others | `(col_sum - self) / (n_fb-1)` | Avg attention received from other blocks |
| 6 | entropy | `-Σ(p·log(p))` | How "spread" this block's attention is |

Cross-layer (3 stats from to_target across all 18 layers):
- mean_to_target, std_to_target, linear_trend_to_target

**Dims**: 18×3×6 + 3 = **327**

#### Enriched mode (enriched=True): 15 stats per layer×head → **813 dims**

Adds 9 more stats (indices 7–15) on top of the basic 6:

| # | Stat | Formula | Interpretation |
|---|------|---------|----------------|
| 7 | max_to_others | `max(outgoing_to_feat_blocks \ self)` | Strongest outgoing attention to any other feature |
| 8 | std_to_others | `std(outgoing \ self)` | Diversity of outgoing attention |
| 9 | max_from_others | `max(incoming_from_feat_blocks \ self)` | Strongest incoming attention from any other feature |
| 10 | std_from_others | `std(incoming \ self)` | Diversity of incoming attention |
| 11 | mean_asymmetry | `mean(out[bi,j] - in[j,bi])` for j≠bi | Whether this block sends more than it receives |
| 12 | max_abs_asymmetry | `max(|out - in|)` | Largest asymmetry with any other block |
| 13 | target_out_rank | `frac(blocks with attn ≤ to_target)` | Percentile rank of to_target among all outgoing attns |
| 14 | target_in_rank | `frac(blocks with attn ≤ from_target)` | Percentile rank of from_target among all incoming attns |
| 15 | contrast_to_target | `to_target - mean_to_others` | How much more this block attends to target vs others |

Cross-layer: same 3 stats.

**Dims**: 18×3×15 + 3 = **813**

#### Attention Rollout (enriched=True only): +15 dims → total **828 dims**

Rollout (Abnar & Zuidema, 2020) computes **cumulative attention flow** across all layers, accounting for residual connections:
```
R_0 = I
R_{l+1} = (0.5·I + 0.5·A_l) @ R_l    (0.5 mixing for skip connection)
```

The `0.5·I + 0.5·A` mixing reflects that at each layer, roughly half the information passes through the residual connection unchanged. After 18 compositions, `R_18[bi, target]` represents the total information flow from feature block `bi` to the target through the entire network.

| # | Stat | Formula | Dims | Interpretation |
|---|------|---------|------|----------------|
| R1 | rollout_to_target | `R_18[bi, target, h]` per head | 3 | Total cumulative flow from this block to target |
| R2 | rollout_from_target | `R_18[target, bi, h]` per head | 3 | Total cumulative flow from target to this block |
| R3 | rollout_self | `R_18[bi, bi, h]` per head | 3 | How much original info this block retains |
| R4 | rollout_to_target_mean | `mean_h(R_18[bi, target])` | 1 | Head-averaged cumulative flow to target |
| R5 | rollout_rank | `frac(blocks with rollout ≤ this)` | 1 | Percentile rank among all blocks |
| R6 | rollout_contrast | `rollout - mean(other blocks)` | 1 | Relative cumulative flow strength |
| R7 | rollout_entropy | `-Σ(R[bi,:] · log(R[bi,:]))` | 1 | How spread the cumulative flow is |
| R8 | rollout_mid | `R_9[bi, target]` mean over heads | 1 | Flow at network midpoint (layer 9) |
| R9 | rollout_early_late_ratio | `R_18 / (R_9 + ε)` | 1 | Whether flow builds early or late |

**Total**: 15 dims. Computed on GPU in ~2.3ms (negligible overhead).

**Why rollout matters**: Per-layer `to_target` captures direct attention at each layer. But in a deep transformer, information flows through **multi-hop paths** — block A may attend to block B in layer 5, and block B attends to target in layer 12. Rollout captures this cumulative flow that no single-layer stat can. `rollout_early_late_ratio` reveals whether the model resolves feature importance early (layers 1–9) or late (layers 10–18).

**Why this matters**: `to_target` and `from_target` directly measure the feature↔target information flow. If TabPFN has learned that feature f3 is causal, the attention from block_1 (containing f3) to the target block should be high. The enriched stats add **relational context**: `contrast_to_target` measures how much a block *preferentially* attends to the target (vs spreading attention everywhere), and `mean_asymmetry` captures whether the feature→target relationship is directional. The `linear_trend` captures whether this attention increases through deeper layers.

### 2.2 Between-Items Attention — INDIRECT SIGNAL

**Raw shape**: `(n_blocks, n_items, n_items, n_heads)` per layer → 18 layers

#### Basic mode (enriched=False): 3 stats per layer×head → **162 dims**

| # | Stat | Formula | Interpretation |
|---|------|---------|----------------|
| 1 | entropy | `mean(entr(rows))` over items | How uniformly items attend to each other in this block |
| 2 | max | `max(flattened)` | Peak attention value (concentration) |
| 3 | variance | `var(flattened)` | How varied the attention pattern is |

**Dims**: 18×3×3 = **162**

#### Enriched mode (enriched=True): 9 stats per layer×head → **486 dims**

Adds 6 train/test split-aware stats (requires `n_train > 0`):

| # | Stat | Formula | Interpretation |
|---|------|---------|----------------|
| 4 | train_to_test | `mean(ah[:, :n_train, n_train:])` | How much train items attend to test items |
| 5 | test_to_train | `mean(ah[:, n_train:, :n_train])` | How much test items attend to train items (ICL signal) |
| 6 | self_train | `mean(ah[:, :n_train, :n_train])` | Train-train self-attention strength |
| 7 | self_test | `mean(ah[:, n_train:, n_train:])` | Test-test self-attention strength |
| 8 | train_test_ratio | `train_to_test / (self_train + ε)` | Relative cross-set vs within-set attention |
| 9 | concentration | `max / mean_row_sum` | How peaked the attention distribution is |

**Dims**: 18×3×9 = **486**

**Why the enriched stats matter**: The basic stats average over all item pairs indiscriminately. The enriched stats decompose by train/test split, capturing the **in-context learning (ICL) pattern**: for a causal feature, test items should attend more to train items (`test_to_train`) because the model needs to transfer learned relationships. The `train_test_ratio` directly measures whether the block facilitates cross-set information flow.

### 2.3 Embeddings — USELESS (identical for all features)

Global stats (mean/std over samples) tiled to every feature. Zero discriminative power.

| Mode | Stats | Dims |
|------|-------|------|
| Both | test_mean(192) + test_std(192) + train_mean(192) | **576** |

All features get the same 576 values via `np.tile()`.

### 2.4 Gradients — MIXED QUALITY

#### Input Gradients (truly per-feature)

**Basic mode**: 4 stats per feature → **4 dims**

| # | Stat | Interpretation |
|---|------|----------------|
| 1 | abs_mean_grad | Average sensitivity of loss to this feature |
| 2 | abs_max_grad | Peak sensitivity |
| 3 | grad_std | Variation in gradient across samples |
| 4 | dominance | Whether gradient is mostly positive or negative |

**Enriched mode**: 8 stats per feature → **8 dims**

| # | Stat | Interpretation |
|---|------|----------------|
| 5 | rank | Percentile rank of this feature's abs_mean among all features |
| 6 | contrast | `abs_mean - mean(other features' abs_mean)` |
| 7 | abs_median | Median |∂L/∂x| (robust to outliers) |
| 8 | grad_energy | Mean of squared gradients |

#### Attention Gradients (per block, requires `extract_gradients=True`)

**Basic mode**: 2 stats per layer×head → 18×3×2 = **108 dims**

| # | Stat | Interpretation |
|---|------|----------------|
| 1 | grad_to_target | Gradient of attention weight from this block to target |
| 2 | mean_abs_grad | Mean |grad| across all destination blocks |

**Enriched mode**: 6 stats per layer×head + 3 cross-layer → 18×3×6 + 3 = **327 dims**

| # | Stat | Interpretation |
|---|------|----------------|
| 3 | grad_from_target | Gradient of attention from target to this block |
| 4 | asymmetry | `grad_to_target - grad_from_target` |
| 5 | max_abs_grad | Peak absolute gradient |
| 6 | rank | Percentile rank of this block's grad_to_target |
| + | cross-layer (3) | trend, mean, std of grad_to_target across layers |

**Note**: In `input_only` extraction mode (v6 default), `retain_grad` is NOT called on attention weights. All attention gradient dicts contain `None`, and the processor fills **zeros** for all attention gradient dims. Only input gradients are populated.

### 2.5 MLP Activations — BLOCK-LEVEL SIGNAL

**Raw shape**: `(batch, n_items, n_blocks, emsize=192)` per layer → 18 layers

#### Basic mode (enriched=False): 5 stats per layer → **90 dims**

| # | Stat | Formula | Interpretation |
|---|------|---------|----------------|
| 1 | mean | `act.mean(axis=emsize)` | Average activation level |
| 2 | std | `act.std(axis=emsize)` | Activation diversity |
| 3 | max | `act.max(axis=emsize)` | Peak activation |
| 4 | sparsity | `frac(|act| < 0.01)` | How many neurons are "off" |
| 5 | l2_norm | `‖act‖₂` | Total activation energy |

**Dims**: 18×5 = **90**

#### Enriched mode (enriched=True): 10 stats per layer + 3 cross-layer → **183 dims**

| # | Stat | Formula | Interpretation |
|---|------|---------|----------------|
| 6 | cosine_to_target | `dot(feat, target) / (‖feat‖·‖target‖)` | Similarity of block's representation to target's |
| 7 | diff_from_mean | `‖feat - mean(all blocks)‖` | How distinctive this block's activations are |
| 8 | min | `act.min(axis=emsize)` | Minimum activation |
| 9 | skewness | `E[(x-μ)³] / σ³` | Asymmetry of activation distribution |
| 10 | pos_frac | `frac(act > 0)` | Fraction of positive activations |

Cross-layer (3 stats):
- **norm_trend**: Linear regression slope of L2 norm across layers
- **cosine_first_last**: Cosine similarity between first and last layer activations
- **norm_ratio**: Last layer norm / first layer norm

**Dims**: 18×10 + 3 = **183**

### 2.6 Items Attention Gradients (full gradient mode only)

Only extracted when `extract_gradients=True` (NOT `"input_only"`). Shape: `(n_blocks, n_items, n_items, n_heads)`.

**6 stats per layer×head** (always, no basic/enriched split):

| # | Stat | Interpretation |
|---|------|----------------|
| 1 | grad_train_to_test | Mean |grad| for train→test item pairs |
| 2 | grad_test_to_train | Mean |grad| for test→train item pairs |
| 3 | grad_self_train | Mean |grad| for train→train item pairs |
| 4 | grad_mean_abs | Overall mean |grad| |
| 5 | grad_max_abs | Peak absolute gradient |
| 6 | concentration | `max / mean` — how peaked the gradient is |

**Dims**: 18×3×6 = **324 dims**

This category is **not available** in `input_only` mode (v6 default).

## 3. Summary: Dimension Counts

### enriched=False (v6 default, 1267 dims)

```
Category                  Dims    Per-Feature?   Quality
─────────────────────────────────────────────────────────
Between-features attn      327    Per-block (3)   ★★★★★  (core: to_target, from_target)
Between-items attn         162    Per-block (3)   ★★☆☆☆  (diluted by item averaging)
Embeddings                 576    NO (identical)  ☆☆☆☆☆  (completely useless)
Input gradients              4    YES (per-feat)  ★★★★★  (direct sensitivity)
Attention gradients        108    ZERO (all 0)*   ☆☆☆☆☆  (*input_only mode → zeros)
MLP activations             90    Per-block (3)   ★★★☆☆  (indirect)
─────────────────────────────────────────────────────────
TOTAL                    1267    Effective: ~583
```

### enriched=True (with rollout)

```
Category                  Dims    Per-Feature?   Quality
─────────────────────────────────────────────────────────
Between-features attn      828    Per-block (3)   ★★★★★  (+asymmetry, ranking, contrast, ROLLOUT)
Between-items attn         486    Per-block (3)   ★★★☆☆  (+train/test split awareness)
Embeddings                 576    NO (identical)  ☆☆☆☆☆  (still useless)
Input gradients              8    YES (per-feat)  ★★★★★  (+rank, contrast, energy)
Attention gradients        327    ZERO (all 0)*   ☆☆☆☆☆  (*input_only mode → zeros)
MLP activations            183    Per-block (3)   ★★★★☆  (+cosine_to_target, skewness)
─────────────────────────────────────────────────────────
TOTAL (no items_attn_grad) 2156†  Effective: ~1253†
```

† With `extract_gradients=True` (full mode), items_attention_gradients adds 324 dims.

### enriched=True + extract_gradients=True (full extraction)

```
Category                  Dims    Notes
─────────────────────────────────────────────────────────
Between-features attn      828    15 stats × 18L × 3H + 3 + 15 rollout
Between-items attn         486    9 stats × 18L × 3H
Embeddings                 576    3 × 192 (tiled)
Input gradients              8    8 per-feature stats
Attention gradients        327    6 stats × 18L × 3H + 3
Items attention gradients  324    6 stats × 18L × 3H
MLP activations            183    10 stats × 18L + 3
─────────────────────────────────────────────────────────
TOTAL                    2732
Effective (excl emb):    2156
```

## 4. Implications and Possible Improvements

### A. Drop Dead Weight (now)
Removing embeddings (576d) and zero attention gradients (108d) reduces input from 1267d → 583d with **zero information loss**. The MLP has fewer parameters, trains faster, and won't waste capacity modeling constants.

### B. The Block Quantization Bottleneck (architectural)
With `features_per_group=3`, at most `ceil(n_features/3)` unique attention patterns exist. For 8 features → only 3 unique patterns. This fundamentally limits how well attention-based features can discriminate individual features.

**Possible mitigations**:
1. **Run TabPFN with features_per_group=1** (if supported): Every feature gets its own block → full per-feature attention resolution. Cost: larger attention matrices.
2. **Feature permutation**: Run TabPFN multiple times with different feature orderings. Feature f0 paired with {f1,f2} in run1 vs {f4,f5} in run2 gives different blocks → richer per-feature signal via aggregation.
3. **Feature ablation gradient**: Instead of d(loss)/d(input), compute d(loss)/d(feature_presence) by running with/without each feature.

### C. Enrich Between-Items Attention (cheap improvement)
Switch from `enriched=False` (3 stats) to `enriched=True` (9 stats) for between-items attention. The additional 6 stats capture train/test interaction patterns that may better differentiate causal vs noise features. Cost: 162 → 486 dims.

### D. Extract Per-Feature Attention (requires model changes)
Instead of block-level stats, extract **element-wise attention within blocks** — e.g., the attention from each of the 3 features in a block to the target, using the raw Q·K computation before the block-level projection. This would require modifying the encoder to expose sub-block information.

## 5. Code References

| Component | File | Key Lines |
|-----------|------|-----------|
| Feature grouping (pad + rearrange) | `src/tabpfn/architectures/base/transformer.py` | 369–395 |
| Target block concatenation | `src/tabpfn/architectures/base/transformer.py` | 515 |
| PerFeatureEncoderLayer | `src/tabpfn/architectures/base/layer.py` | 99–430 |
| Between-features attn call | `src/tabpfn/architectures/base/layer.py` | 306–313 |
| Between-items attn call (with transpose) | `src/tabpfn/architectures/base/layer.py` | 315–368 |
| Attention weight computation | `src/tabpfn/architectures/base/attention/full_attention.py` | 657–798 |
| Attention weight caching | `src/tabpfn/architectures/base/attention/full_attention.py` | 564–571 |
| Signal extraction (hooks + collection) | `src/tabpfn/interpretation/extraction/signal_extractor.py` | 54–407 |
| Between-features stats processing | `src/tabpfn/interpretation/extraction/signal_processor.py` | 340–478 |
| Between-items stats processing | `src/tabpfn/interpretation/extraction/signal_processor.py` | 480–555 |
| GPU stats computation | `src/tabpfn/interpretation/extraction/gpu_stats_computer.py` | full file |
| NormalizeFeatureGroups encoder step | `src/tabpfn/architectures/encoders/steps/normalize_feature_groups_encoder_step.py` | full file |
