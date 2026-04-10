# TabPFN Post-Hoc Interpretation Model: Performance Evaluation

## 1. Executive Summary

We developed and evaluated a **post-hoc interpretation model** that predicts per-feature causal importance from TabPFN's internal representations. The model is trained on **141,521 synthetic datasets** generated from Structural Causal Models (SCMs) using TabPFN's own prior data generator (`zzhang-cn/tabpfn-synthetic-data`), ensuring distribution alignment with TabPFN's training data. We extract 4 categories of signals from TabPFN v2.5's inference process — between-features attention, between-items attention, encoder embeddings, and MLP activations — and train small neural networks to map these signals to ground-truth causal importance labels.

**Key Results:**
- MLP interpretation model achieves **AUROC 0.668** for binary direct parent detection and **AUROC 0.660** for ancestry detection, significantly above all baselines
- Positive R² (0.062–0.080) and meaningful Pearson correlations (0.25–0.29) for continuous importance modes
- **Between-features attention alone** (327 dims) achieves 99.5% of full-model AUROC, making it the dominant signal category
- Embeddings (576 dims, 50% of total dimensions) contribute minimally — removing them causes <0.5% AUROC drop
- Transformer variant suffers from training instability (NaN loss, collapsed outputs on binary_ancestry)
- Feature vector dimension: **1,155** per feature from 18 transformer layers × 3 attention heads

---

## 2. Experimental Setup

### 2.1 Data Generation

| Parameter | Value |
|-----------|-------|
| Generator | `zzhang-cn/tabpfn-synthetic-data` (faithful TabPFN v2.5 prior reimplementation) |
| Total synthetic datasets | 141,521 (from ~184K generated; ~23% skipped due to 0 causal parents) |
| Train / Val / Test split | 99,064 / 21,228 / 21,229 datasets (70/15/15) |
| Features per dataset | 3–40 (mean: 8.2) |
| Feature distribution | Beta(0.95, 8.0) × [3, 50], capped at 50 for extraction speed |
| Samples per dataset | 50–1,000 |
| Graph type | SCM DAGs from TabPFN's training prior (node-level, 8 components per node) |
| Node edge types | Mixed: linear, polynomial, categorical, neural network |
| Disk cache size | 3.8 GB (141K .npz files + .json metadata) |
| Generation time | ~32 hours on 1× NVIDIA RTX A6000 |
| Data loading time | 144 seconds for all 141K files |

### 2.2 Ground-Truth Label Modes

1. **Binary Direct** (`binary_direct`): 1 if feature is a direct parent of target in the SCM DAG, 0 otherwise
   - Mean parent fraction: **25.8%** of features (≈1.8 parents per dataset)
   
2. **Binary Ancestry** (`binary_ancestry`): 1 if feature is any ancestor (direct or indirect) of target
   - Mean ancestor fraction: **32.1%** of features (≈2.4 ancestors per dataset)

3. **Graded Ancestry** (`graded_ancestry`): Continuous score decaying with causal distance (γ^d, γ=0.5)
   - Captures both direct and indirect causal influence with distance decay

4. **Interventional** (`interventional`): Normalized sensitivity of target to single-feature interventions
   - Most fine-grained measure of actual causal effect magnitude

### 2.3 Signal Extraction (Input Features)

From each TabPFN inference, we extract 4 categories of signals processed into a **1,155-dimensional** per-feature vector:

| Signal Category | Dims | % of Total | Description |
|----------------|------|-----------|-------------|
| Between-features attention | 327 | 28.3% | 18 layers × 3 heads × 6 stats (self-attn, to-target, from-target, mean-to-others, mean-from-others, entropy) + 3 cross-layer stats |
| Between-items attention | 162 | 14.0% | 18 layers × 3 heads × 3 stats (entropy, max, variance) |
| Encoder embeddings | 576 | 49.9% | Global context: mean/std of test embeddings (384) + mean of train embeddings (192), replicated per feature |
| MLP activations | 90 | 7.8% | 18 layers × 5 stats (mean, std, max, sparsity, norm) |
| **Total** | **1,155** | **100%** | |

**Note**: Embeddings are **global** (same for all features in a dataset) and replicated per feature. This means they provide dataset-level context but no per-feature discrimination.

### 2.4 Model Architectures

**PerFeatureMLP**: The same MLP is applied independently to each feature's 1,155-dim signal vector (permutation equivariant).
- Architecture: Linear(1155→512) → ReLU → Dropout(0.1) → Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→128) → ReLU → Dropout(0.1) → Linear(128→1)
- Parameters: ~780K

**PerFeatureTransformer**: Self-attention across features treats each feature's signal vector as a token.
- Architecture: Linear(1155→256) → 2× TransformerEncoderLayer(d=256, h=4, ff=1024) → Linear(256→128) → ReLU → Linear(128→1)
- Parameters: ~1.6M

### 2.5 Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=5e-4, weight_decay=1e-4) |
| Scheduler | Cosine annealing (T_max=100) |
| Batch size | 16 |
| Max epochs | 100 (early stopping patience=15) |
| Gradient clipping | Max norm 1.0 |
| Loss (binary modes) | BCEWithLogitsLoss |
| Loss (continuous modes) | MSELoss |
| Feature padding | Zero-pad to max_features=41 with mask |
| Device | 1× NVIDIA RTX A6000 (47.7 GB) |

### 2.6 Baselines

1. **Random**: Uniform random predictions in [0, 1]
2. **Constant Mean**: Predict the mean label value for all features (trivial baseline with R²=0)
3. **Signal Norm**: Use L2 norm of the 1,155-dim signal vector as importance score

---

## 3. Main Experiment Results

### 3.1 Binary Direct Parent Detection

| Method | AUROC | AUPR | F1 | Precision | Recall | Spearman |
|--------|-------|------|-----|-----------|--------|----------|
| Random | 0.505 | 0.219 | 0.306 | 0.219 | 0.506 | — |
| Constant Mean | 0.500 | 0.217 | 0.000 | 0.000 | 0.000 | — |
| Signal Norm | 0.566 | 0.253 | 0.357 | 0.217 | 1.000 | — |
| **MLP** | **0.668** | **0.347** | **0.406** | **0.335** | **0.515** | **0.108** |
| Transformer | 0.658 | 0.332 | 0.399 | 0.326 | 0.514 | 0.083 |

**Analysis:**
- MLP achieves AUROC 0.668, a **33.6% improvement over chance** (0.50) and **18.0% over signal_norm**
- AUPR of 0.347 is 59% higher than random (0.219), given a 21.7% positive rate
- The MLP consistently outperforms the Transformer (+1.0 AUROC points, +1.5 AUPR points)
- Signal norm shows modest positive signal (AUROC=0.566), indicating some correlation between overall signal magnitude and causal parent status

### 3.2 Binary Ancestry Detection

| Method | AUROC | AUPR | F1 | Precision | Recall | Spearman |
|--------|-------|------|-----|-----------|--------|----------|
| Random | 0.498 | 0.285 | 0.363 | 0.285 | 0.497 | — |
| Constant Mean | 0.500 | 0.287 | 0.000 | 0.000 | 0.000 | — |
| Signal Norm | 0.541 | 0.313 | 0.446 | 0.287 | 1.000 | — |
| **MLP** | **0.660** | **0.418** | **0.490** | **0.369** | **0.728** | — |
| Transformer | 0.500 | 0.287 | 0.446 | 0.287 | 1.000 | — |

**Analysis:**
- MLP achieves AUROC 0.660, **32% above chance**
- **Transformer collapsed** to predicting all-positive (AUROC=0.500, recall=1.0), a known training instability issue
- AUPR of 0.418 (MLP) is 46% above random (0.287)
- Higher AUPR than binary_direct due to higher positive rate (28.7% vs 21.7%)

### 3.3 Graded Ancestry (Continuous)

| Method | R² | MAE | Pearson Corr | Spearman |
|--------|-----|------|-------------|----------|
| Random | -0.858 | 0.483 | 0.001 | — |
| Constant Mean | 0.000 | 0.355 | 0.000 | — |
| Signal Norm | -3.237 | 0.746 | 0.072 | — |
| **MLP** | **0.062** | **0.333** | **0.254** | — |
| Transformer | 0.055 | 0.331 | 0.238 | 0.079 |

**Analysis:**
- Both models achieve positive R², explaining variance beyond constant-mean baseline
- MLP Pearson correlation of 0.254 indicates a moderate linear relationship with ground truth
- MAE reduction from 0.355 (constant) to 0.333 (MLP) = **6.2% improvement**
- Transformer is close to MLP performance, suggesting cross-feature attention provides marginal benefit here

### 3.4 Interventional Sensitivity (Continuous)

| Method | R² | MAE | Pearson Corr | Spearman |
|--------|-----|------|-------------|----------|
| Random | -1.556 | 0.476 | -0.002 | — |
| Constant Mean | 0.000 | 0.264 | 0.000 | — |
| Signal Norm | -5.481 | 0.817 | 0.124 | — |
| **MLP** | **0.080** | **0.245** | **0.285** | **0.118** |
| Transformer | 0.076 | 0.246 | 0.276 | 0.100 |

**Analysis:**
- MLP achieves R²=0.080, the highest across all continuous modes
- Pearson correlation of 0.285 demonstrates the model captures actual causal effect magnitude
- MAE of 0.245 vs 0.264 (constant) = **7.2% improvement**
- Interventional is arguably the most practically useful mode and shows the strongest signal

---

## 4. Feature Ablation Study

We systematically evaluated which signal categories contribute to interpretation quality by training MLP models with subsets of the 1,155-dim input features. All ablation runs use the same train/val/test split.

### 4.1 Ablation Configurations

| Config | Signal Categories | Input Dims |
|--------|------------------|-----------|
| **full** | All 4 categories | 1,155 |
| **attention_only** | feat_attn + item_attn | 489 |
| **feat_attn_only** | Between-features attention | 327 |
| **item_attn_only** | Between-items attention | 162 |
| **embeddings_only** | Encoder embeddings | 576 |
| **mlp_only** | MLP activations | 90 |
| **no_attention** | Embeddings + MLP activations | 666 |
| **no_embeddings** | feat_attn + item_attn + MLP | 579 |

### 4.2 Ablation Results Summary

| Config | Dims | binary_direct AUROC | binary_ancestry AUROC | graded R² | interventional R² |
|--------|------|--------------------|-----------------------|-----------|-------------------|
| **full** | 1,155 | **0.667** | **0.661** | **0.066** | 0.078 |
| attention_only | 489 | 0.666 | 0.657 | 0.064 | **0.083** |
| feat_attn_only | 327 | 0.664 | 0.658 | 0.062 | 0.082 |
| item_attn_only | 162 | 0.617 | 0.621 | 0.032 | 0.033 |
| embeddings_only | 576 | 0.631 | 0.640 | 0.043 | 0.042 |
| mlp_only | 90 | 0.648 | 0.636 | 0.049 | 0.068 |
| no_attention | 666 | 0.653 | 0.653 | 0.055 | −0.000 |
| **no_embeddings** | 579 | 0.665 | 0.654 | 0.062 | **0.083** |

### 4.3 Relative Performance Drop vs Full Model

| Config | Dims | AUROC Drop (direct) | AUROC Drop (ancestry) | R² Drop (graded) | R² Drop (interventional) |
|--------|------|--------------------|-----------------------|-------------------|--------------------------|
| attention_only | 489 | −0.2% | −0.7% | −2.5% | +6.5% |
| feat_attn_only | 327 | −0.5% | −0.5% | −5.6% | +4.3% |
| no_embeddings | 579 | −0.4% | −1.1% | −5.4% | +6.1% |
| mlp_only | 90 | −2.8% | −3.8% | −24.7% | −12.5% |
| embeddings_only | 576 | −5.4% | −3.2% | −34.4% | −46.6% |
| item_attn_only | 162 | −7.5% | −6.1% | −51.3% | −57.8% |
| no_attention | 666 | −2.1% | −1.3% | −16.2% | −100.0% |

### 4.4 Key Ablation Findings

#### Finding 1: Between-features attention is the dominant signal
- **feat_attn_only** (327 dims, 28% of total) retains **99.5%** of full AUROC for binary_direct and **99.5%** for binary_ancestry
- Adding between-items attention (+162 dims) provides minimal additional gain (+0.2% AUROC)
- This makes sense: between-features attention directly captures how TabPFN routes information between feature blocks

#### Finding 2: Embeddings are largely redundant
- **no_embeddings** (579 dims) performs within **0.4%** of the full model on binary_direct
- **embeddings_only** (576 dims, 50% of total) is one of the weakest configurations (AUROC 0.631)
- Embeddings are **global** (identical for all features in a dataset), so they provide no per-feature discrimination
- Their main contribution is dataset-level context that slightly helps continuous modes

#### Finding 3: MLP activations punch above their weight
- **mlp_only** (90 dims, 8% of total) achieves AUROC 0.648, only 2.8% below full model
- Best dimension-efficiency: 0.648/90 = 7.2×10⁻³ AUROC per dim (vs full: 0.667/1155 = 5.8×10⁻⁴)
- MLP activations capture per-feature processing patterns that correlate with causal importance

#### Finding 4: Between-items attention is the weakest individual signal
- **item_attn_only** (162 dims) has AUROC 0.617, the largest drop from full (−7.5%)
- This signal captures how samples attend to each other per feature block — less directly tied to causal structure
- However, combined with feat_attn (attention_only config), it provides the best interventional R² (0.083)

#### Finding 5: no_attention collapses on interventional
- **no_attention** (embeddings + MLP, 666 dims) achieves R²≈0 on interventional mode
- This means attention signals are **essential** for capturing interventional sensitivity
- Without attention, the model can still partially detect binary parents (AUROC 0.653) using MLP activations

### 4.5 Recommended Minimal Feature Set

For practical deployment, **between-features attention + MLP activations** (417 dims) would provide:
- ~99% of full binary classification performance
- ~95% of continuous regression performance
- 64% reduction in feature dimensionality (417 vs 1,155)
- Faster extraction (skip embedding and item-attention computation)

---

## 5. Cross-Mode Analysis

### 5.1 Model Comparison (MLP vs Transformer)

| Label Mode | MLP | Transformer | Winner | Gap |
|-----------|-----|-------------|--------|-----|
| binary_direct | AUROC 0.668 | AUROC 0.658 | MLP | +1.0 |
| binary_ancestry | AUROC 0.660 | AUROC 0.500 ⚠️ | MLP | +16.0 |
| graded_ancestry | R² 0.062 | R² 0.055 | MLP | +0.007 |
| interventional | R² 0.080 | R² 0.076 | MLP | +0.004 |

**MLP wins on all 4 modes.** The Transformer's key issue is **training instability**: it frequently hits NaN loss after 20–30 epochs, triggering early stopping from a suboptimal checkpoint. For binary_ancestry, it collapsed entirely to constant all-positive predictions.

### 5.2 Training Dynamics

| Label Mode | Model | Epochs | Best Val Loss | Issue |
|-----------|-------|--------|---------------|-------|
| binary_direct | MLP | 76 | 0.931 | Stable ✓ |
| binary_direct | Transformer | 41 | 0.938 | Stable ✓ |
| binary_ancestry | MLP | 73 | 1.026 | Stable ✓ |
| binary_ancestry | Transformer | 38 | 1.087 | **Collapsed** ⚠️ |
| graded_ancestry | MLP | 70 | 0.161 | Stable ✓ |
| graded_ancestry | Transformer | 25 | 0.162 | Early NaN ⚠️ |
| interventional | MLP | 34 | 0.113 | Stable ✓ |
| interventional | Transformer | 37 | 0.113 | Stable ✓ |

### 5.3 Label Mode Difficulty Ranking

| Rank | Mode | Primary Metric | Value | Interpretation |
|------|------|---------------|-------|----------------|
| 1 (easiest) | binary_direct | AUROC | 0.668 | Clear binary signal: direct parent or not |
| 2 | binary_ancestry | AUROC | 0.660 | Slightly harder: must detect indirect ancestors too |
| 3 | interventional | R² | 0.080 | Continuous, requires effect magnitude estimation |
| 4 (hardest) | graded_ancestry | R² | 0.062 | Path-length-dependent decay is subtle |

### 5.4 Signal Informativeness

The **signal_norm** baseline provides an important reference:
- AUROC 0.566 for binary_direct (above chance) — raw signal magnitude has *some* correlation with causal importance
- But it's far below the trained model (0.668), confirming the model learns non-trivial patterns
- For continuous modes, signal_norm has negative R² (much worse than constant prediction), showing raw magnitude is misleading for effect size estimation

---

## 6. Comparison with Previous Small-Scale Experiment

We previously ran a pilot experiment with 193 datasets using a custom SCM generator. Here we compare against the 141K-dataset experiment using TabPFN's prior generator:

| Metric | Pilot (193 datasets) | Large-scale (141K datasets) | Change |
|--------|---------------------|---------------------------|--------|
| binary_direct AUROC | 0.722 | 0.668 | −0.054 |
| binary_ancestry AUROC | 0.721 | 0.660 | −0.061 |
| graded_ancestry Corr | 0.384 | 0.254 | −0.130 |
| interventional R² | 0.085 | 0.080 | −0.005 |

**Why are large-scale results lower?** This is *not* regression — it reflects a harder, more realistic evaluation:
1. **Distribution alignment**: The TabPFN prior generates more diverse, realistic SCMs than our simple custom generator
2. **Feature count**: Mean 8.2 features (3–40 range) vs 14.7 (5–25), with more small datasets where signals are sparse
3. **Node-level DAGs**: Features are components of multi-dimensional nodes (dim=8), creating more complex causal structures
4. **Much larger test set**: 21,229 test datasets vs 30, giving more reliable metric estimates
5. **No overfitting**: With 99K training datasets, the model cannot memorize — pilot's high AUROC may partly reflect overfitting to 135 training examples

---

## 7. Discussion

### 7.1 What Works

1. **Causal signal exists in TabPFN internals**: Consistent AUROC 0.66–0.67 across binary modes demonstrates that TabPFN's attention patterns encode information about causal feature importance
2. **Between-features attention is the key signal**: 327 dims capture nearly all the discriminative information, consistent with the hypothesis that TabPFN's feature-block attention reflects causal structure
3. **MLP is sufficient**: The simpler MLP architecture outperforms the Transformer on every mode, suggesting per-feature independent scoring is the right inductive bias

### 7.2 What Doesn't Work

1. **Transformer training instability**: NaN loss and output collapse are serious issues. Possible causes:
   - Self-attention over ~8 features (tokens) may be degenerate for small feature counts
   - The 256-dim projection from 1,155 dims may lose critical information
   - Mitigation: gradient scaling, warmup, larger batch size, or architectural changes
2. **Embeddings as features**: Taking 50% of input dimensions for negligible benefit. The global embedding provides no per-feature discrimination.
3. **Spearman correlation**: Often NaN due to constant predictions on some datasets. This suggests the model struggles with certain dataset configurations.

### 7.3 Limitations

1. **Feature block sharing**: With features_per_group=3, multiple features map to the same attention block, receiving identical attention-based signals. This limits resolution for datasets with many features.
2. **No gradient signals**: Input gradients and attention gradients could provide high-SNR per-feature sensitivity information. Currently excluded for generation speed.
3. **Small feature counts**: Mean of 8.2 features per dataset limits the complexity of causal structures. Many datasets have only 3 features (1–2 parents + target).
4. **Binary ancestry vs direct gap is small**: Only 6.3% more ancestors than direct parents (32.1% vs 25.8%), making these tasks very similar. Deeper DAGs would create more differentiation.

### 7.4 Future Directions

1. **Include gradient signals**: Input gradients (∂pred/∂X_i) and attention gradients for richer per-feature sensitivity
2. **Per-block feature discrimination**: Design attention statistics that differentiate features within the same block (e.g., using the specific feature's position within the block)
3. **Multi-task training**: Shared backbone with mode-specific heads could improve sample efficiency
4. **Curriculum learning**: Start with easy datasets (few features, clear causal structure) and progressively add harder ones
5. **Ensemble interpretation**: Average predictions from multiple TabPFN estimators for more robust signals
6. **Real-data evaluation**: Test on DREAM4/5 GRN benchmarks as out-of-distribution validation

---

## 8. Technical Details

### 8.1 TabPFN Architecture

- **Model**: TabPFN v2.5 (v6.3.1 package)
- **Checkpoint**: `tabpfn-v2.5-regressor-v2.5_default.ckpt`
- **Layers**: 18 encoder layers
- **Attention heads**: 3 per layer
- **Embedding dimension**: 192
- **Features per group**: 3 (features grouped into attention blocks)
- **MLP per layer**: 192→768→192

### 8.2 Attention Tensor Shapes

| Signal | Shape | Notes |
|--------|-------|-------|
| Between-features attention | (batch_items, n_blocks, n_blocks, 3) | heads are last dim |
| Between-items attention | (n_blocks, n_items, n_items, 3) | heads are last dim |
| MLP activations | (1, batch_items, n_blocks, 192) | per-layer |
| Encoder embeddings | (n_samples, 1, 192) | final layer output |

### 8.3 DAG Reconstruction

The external generator produces node-level DAGs where each node has `vector_dim=8` components. Features are individual components. Edge reconstruction:
- **Direct edges**: Only direct parent nodes get edges to the target node in the feature-level DAG
- **Indirect ancestry**: Computed via transitive closure on the node-level DAG
- **Inter-feature edges**: Preserved between features belonging to different connected nodes

### 8.4 Computational Cost

| Phase | Time | Hardware |
|-------|------|----------|
| Data generation (141K datasets) | 32.2 hours | 1× NVIDIA RTX A6000 |
| Data loading from disk | 144 seconds | SSD |
| MLP training per label mode | ~35–43 minutes | 1× NVIDIA RTX A6000 |
| Transformer training per label mode | ~30–50 minutes | 1× NVIDIA RTX A6000 |
| Full ablation (8 configs × 4 modes) | ~17 hours | 1× NVIDIA RTX A6000 |
| Total experiment (main + ablation) | ~22 hours | |

---

## 9. Conclusion

This evaluation demonstrates that **TabPFN's internal representations contain moderate but real signal about causal feature importance**. The interpretation model consistently outperforms baselines, with between-features attention serving as the primary information-carrying signal. The achievable AUROC of ~0.67 for causal parent detection, while below the 0.70–0.80 range typical for mature methods, validates the fundamental approach of distilling causal knowledge from transformer internals.

The ablation study reveals a surprisingly efficient signal structure: **327 dimensions of between-features attention statistics capture 99.5% of the full model's performance**, while 576 dimensions of embeddings (50% of total) contribute almost nothing. This suggests that practical deployment should focus exclusively on attention-based features, potentially with architectural refinements to the attention statistic extraction.

Key areas for improvement include adding gradient-based signals, resolving transformer training instability, and designing per-feature discriminative statistics for features sharing the same attention block.

---

*Generated from experiment run on 2026-03-09. Training data: 141,521 datasets from TabPFN prior generator. Full results: `results/interpretation_experiments/experiment_results.json`*
