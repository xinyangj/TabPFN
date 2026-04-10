# TabPFN Interpretation Feature Design Report

## Overview

This report documents the feature extraction pipeline used by the TabPFN interpretation model — a post-hoc method that predicts per-feature causal importance from TabPFN's internal representations. Features are extracted by hooking into TabPFN's forward pass and computing summary statistics from attention patterns, gradients, MLP activations, and token embeddings.

**TabPFN v2.5 Architecture:**
- 18 transformer layers, 3 attention heads, embedding dimension = 192
- Features grouped into blocks (features_per_group = 3)
- Two attention mechanisms per layer: between-features and between-items
- Target variable treated as an additional feature block

**Feature Versions:**

| Version | Categories | Total Dims | Training Data |
|---------|-----------|-----------|---------------|
| v2 | feat_attn + items_attn + gradients + mlp | 691 | ~141K synthetic |
| v6 | feat_attn + items_attn + **embeddings** + gradients + mlp | 1267 | 524K synthetic |
| Slim | feat_attn + items_attn + grad[0:4] + mlp | 583 | various |

---

## 1. Between-Features Attention (327 dims)

### What it captures
How TabPFN distributes attention **between input features** (treated as blocks) at each transformer layer. This is the core signal — it reveals which features TabPFN considers most relevant when predicting a target.

### How it's extracted
TabPFN groups input features into blocks of 3 and adds a target block. The between-features self-attention operates on these blocks: `Q, K, V ∈ (batch_items, n_blocks, emsize)`. We hook into each layer's `self_attn_between_features` to capture the full attention matrix, then average over the batch_items (data samples) dimension to get a single `(n_blocks, n_blocks)` matrix per layer per head.

### Statistics computed (6 per head per layer)
For each feature block, at each of the 18 layers × 3 heads:

| Stat | Description |
|------|-------------|
| **self_attention** | Diagonal element — how much a feature attends to itself |
| **to_target** | Attention from this feature to the target block |
| **from_target** | Attention from the target block to this feature |
| **mean_to_others** | Average attention to all other non-target features |
| **mean_from_others** | Average attention from all other non-target features |
| **entropy** | Shannon entropy of this feature's attention distribution |

Plus 3 cross-layer aggregation stats on the `to_target` signal:
- **mean** across layers
- **std** across layers
- **trend** (correlation between layer index and to_target value)

### Dimension formula
```
18 layers × 3 heads × 6 stats + 3 cross-layer = 324 + 3 = 327
```

### Importance
This is the most informative single category. In ablation studies, Feature Attention Only (327d) achieves AUROC = 0.664 — only 0.003 below the full model (0.667). The simple "attention heuristic" (sum of these 327 values) alone achieves AUROC = 0.587 on DREAM4-10.

---

## 2. Between-Items Attention (162 dims)

### What it captures
How TabPFN distributes attention **across data samples** (train + test items) for each feature block. This captures whether certain features cause TabPFN to weight specific training examples more heavily.

### How it's extracted
The between-items self-attention operates on data points: `Q, K, V ∈ (n_blocks, n_items, emsize)` where `n_items = n_train + n_test`. For each feature block, we get a `(n_items, n_items)` attention matrix per layer per head. We compute distributional statistics over the items dimension.

### Statistics computed (3 per head per layer)
For each feature block:

| Stat | Description |
|------|-------------|
| **entropy** | Shannon entropy of item attention distribution — high entropy means uniform attention |
| **max** | Maximum attention weight — indicates if attention concentrates on specific items |
| **variance** | Variance of attention weights — measures spread of attention |

### Dimension formula
```
18 layers × 3 heads × 3 stats = 162
```

### Importance
Items attention alone is the weakest single category (AUROC = 0.617), but it provides complementary signal. Combining both attention types (Feature + Items = 489d) yields AUROC = 0.666, only 0.001 below the full 691d model.

---

## 3. Embeddings (576 dims) — v6 only

### What it captures
The intermediate token embeddings from the transformer encoder, summarized across all data samples. These represent a **global dataset-level representation**, not per-feature information.

### How it's extracted
A hook on `transformer_encoder` captures the embedded input tensor of shape `(n_samples, 1, emsize=192)`. We split by train/test and compute:

| Component | Description | Dims |
|-----------|-------------|------|
| **test_mean** | Mean embedding across test samples | 192 |
| **test_std** | Std of embeddings across test samples | 192 |
| **train_mean** | Mean embedding across train samples | 192 |

### Dimension formula
```
3 × 192 = 576
```

### Critical limitation
**These 576 dimensions are identical for all features in a dataset.** They provide global context (e.g., dataset complexity, sample size effects) but cannot distinguish which features are important. This makes them theoretically uninformative for per-feature importance prediction.

### Importance
Embeddings Only achieves AUROC = 0.631 on synthetic data — above random (0.505) because they capture dataset-level biases correlated with feature importance distributions. However, removing embeddings barely hurts the full model (0.665 vs 0.667, only −0.002 AUROC). On DREAM4-10, removing embeddings and normalization actually **improves** transfer performance (v6_slim 0.562 vs v6_full 0.495), suggesting they cause overfitting to synthetic distribution patterns.

---

## 4. Gradients (112 dims)

### What it captures
How the loss landscape responds to each feature, measured through two complementary gradient signals:
1. **Input gradients**: `∂loss/∂x_i` — direct sensitivity of the loss to each feature's values
2. **Attention gradients**: `∂loss/∂α_ij` — how the loss changes with attention weight changes

### How it's extracted
**Input gradients (4 dims):** During a forward-backward pass with a regression loss on test predictions, we compute `∂loss/∂X_input` where `X_input` has shape `(n_samples, n_features)`. For each feature, we summarize the gradient vector across samples.

**Attention gradients (108 dims):** We also capture `∂loss/∂attention_weights` for the between-features attention at each layer. This gives gradient flow information through the attention mechanism.

### Statistics computed

**Input gradient stats (4 per feature):**

| Stat | Description |
|------|-------------|
| **abs_mean_grad** | Mean of |∂loss/∂x_i| across samples |
| **abs_max_grad** | Maximum |∂loss/∂x_i| across samples |
| **grad_std** | Standard deviation of gradients |
| **dominance** | max(frac_positive, frac_negative) — gradient sign consistency |

**Attention gradient stats (2 per head per layer):**

| Stat | Description |
|------|-------------|
| **grad_to_target** | Gradient of attention weight from this feature to target |
| **mean_abs_grad** | Mean absolute gradient across all attention positions |

### Dimension formula
```
4 (input) + 18 layers × 3 heads × 2 stats (attention) = 4 + 108 = 112
```

### Importance
Gradient features are informative but noisy. Adding full gradients without BatchNorm drops AUROC from 0.669 to 0.618. With BatchNorm, gradients recover to 0.666. The slim model variant uses only the 4 input gradient stats (dropping the 108 attention gradient dims) and retains 98.5% of synthetic performance while improving DREAM4 transfer.

---

## 5. MLP Activations (90 dims)

### What it captures
The activation patterns in each transformer layer's feed-forward network, summarized per feature block. These capture how each feature is transformed through the network's nonlinear layers.

### How it's extracted
Forward hooks on each layer's MLP module capture the output tensor of shape `(batch, n_items, n_blocks, emsize=192)`. We average over batch and items dimensions to get a `(n_blocks, 192)` tensor per layer, then compute summary statistics per feature block.

### Statistics computed (5 per layer)
For each feature block at each of the 18 layers:

| Stat | Description |
|------|-------------|
| **mean** | Mean activation value across the 192 embedding dims |
| **std** | Standard deviation of activations |
| **max** | Maximum activation value |
| **sparsity** | Fraction of activations with |value| < 0.01 |
| **L2_norm** | Euclidean norm of the activation vector |

### Dimension formula
```
18 layers × 5 stats = 90
```

### Importance
MLP Only is surprisingly strong at AUROC = 0.648 (vs 0.667 full), making it the second-best single category after feature attention. It captures complementary information about how features are processed through the network's depth.

---

## Performance Summary

### Category Ablation (In-Domain Synthetic, v2 features, ~141K datasets)

| Feature Set | Dims | AUROC | AUPR | F1 |
|-------------|------|-------|------|-----|
| **Full (v2)** | 691 | **0.667** | 0.347 | 0.408 |
| Attention Only | 489 | 0.666 | 0.351 | 0.405 |
| Feature Attn Only | 327 | 0.664 | 0.340 | 0.403 |
| MLP Only | 90 | 0.648 | 0.324 | 0.378 |
| No Embeddings | 579 | 0.665 | 0.345 | 0.398 |
| Embeddings Only | 576 | 0.631 | 0.312 | 0.370 |
| Item Attn Only | 162 | 0.617 | 0.300 | 0.360 |
| No Attention | 666 | 0.653 | 0.314 | 0.399 |

### Cross-Domain Transfer (DREAM4-10, 10 genes × 5 networks)

| Model | Dims | Training Data | AUROC | AUPR |
|-------|------|--------------|-------|------|
| Random Forest | 691 | — | **0.719** | **0.364** |
| Gradient Boosting | 691 | — | 0.682 | 0.325 |
| v2 model | 691 | v2 ~141K | 0.617 | 0.289 |
| DREAM-matched slim | 583 | DREAM-match 23K | 0.606 | 0.243 |
| Attention heuristic | 327 | — | 0.587 | 0.226 |
| v6 slim (no norm) | 583 | v6 524K | 0.562 | — |
| v6 full (with norm) | 1267 | v6 524K | 0.495 | 0.205 |
| Random | — | — | 0.542 | 0.220 |

### In-Domain Synthetic (Various Training Configurations)

| Model | Dims | Training Data | Synthetic AUROC |
|-------|------|--------------|----------------|
| v6_600k_p30 | 1267 | v6 524K | **0.791** |
| v6_600k_deep | 1267 | v6 524K | 0.791 |
| v6_600k | 1267 | v6 524K | 0.791 |
| v6_slim_deep (no norm) | 583 | v6 524K | 0.789 |
| v6_300k | 1267 | v6 268K | 0.785 |
| v2 | 691 | v2 ~141K | 0.676 |
| DREAM-matched slim | 583 | DREAM 23K | 0.620 |

---

## Key Findings

1. **Feature attention dominates**: 327d of feature attention alone captures 99.5% of full model performance on synthetic data. The remaining categories provide marginal gains.

2. **Embeddings are uninformative per-feature**: Being identical across features, they encode dataset-level biases that help on synthetic data but hurt transfer. Removing them costs only −0.002 AUROC in-domain but gains +0.067 AUROC on DREAM4-10.

3. **Attention gradients are noisy**: The 108 attention gradient dims require careful normalization (BatchNorm) to avoid hurting performance. The 4 input gradient dims are cleaner and sufficient for the slim model.

4. **In-domain ≠ transfer performance**: v6 models achieve 0.791 AUROC on synthetic data but only 0.495 on DREAM4-10 — worse than random. The v2 model with simpler features (0.676 synthetic) transfers much better (0.617 DREAM4-10).

5. **Distribution matching > dataset size**: Training on 23K DREAM-matched samples (0.606 DREAM4-10) nearly matches training on 141K v2 samples (0.617), demonstrating that matching the target domain's feature count distribution is more important than scale.

6. **RF baseline gap**: Random Forest trained directly on extracted features achieves 0.719 AUROC on DREAM4-10, suggesting the learned MLP doesn't fully exploit the feature representations. The bottleneck may be the training distribution rather than the features themselves.
