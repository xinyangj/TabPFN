# GRN Evaluation Infrastructure Summary - TabPFN

## Overview
The TabPFN codebase has a comprehensive Gene Regulatory Network (GRN) inference infrastructure that uses TabPFN as a frozen foundation model to predict regulatory relationships between transcription factors (TFs) and target genes. **Important**: No fine-tuning occurs—TabPFN's weights are frozen; only in-context learning is used.

---

## 1. GRN Benchmark Datasets

### DREAM4 (In Silico Networks)
**Location**: `/home/xinyangjiang/Projects/TabPFN/data/dream4/dream4/`

**Structure**: Per-network files with naming pattern `dream4_{SIZE}_net{ID}_{TYPE}.{EXT}`
- **Sizes**: 10, 50, 100 genes
- **Networks**: 5 networks per size (ID 1-5)
- **File formats**:
  - `dream4_{size}_net{id}_expression.npy` - Gene expression matrix (n_samples × n_genes)
  - `dream4_{size}_net{id}_genes.csv` - Gene names
  - `dream4_{size}_net{id}_tfs.csv` - Transcription factor names
  - `dream4_{size}_net{id}_gold_standard.csv` - Ground truth regulatory edges

**Example**: `/data/dream4/dream4/dream4_10_net1_expression.npy`

### DREAM5 (Real Biological Networks)
**Location**: `/home/xinyangjiang/Projects/TabPFN/data/dream5/`

**Datasets**:
- **E. coli**: 4,511 genes, 334 TFs, 806 samples
- **S. cerevisiae**: Real yeast data
- **S. aureus**: Real staphylococcus data

**File naming**: `{organism}_expression.{npy|csv}`, `{organism}_genes.{csv|txt}`, `{organism}_tfs.{csv|txt}`, `{organism}_gold_standard.csv`

**Fallback**: If files don't exist, `DREAMChallengeLoader` generates synthetic data with warnings (for testing).

### Data Format Details
```python
expression: np.ndarray
  Shape: (n_samples, n_genes)
  Type: float32
  Interpretation: Gene expression levels across samples

gold_standard: pd.DataFrame or set[tuple[str, str]]
  DataFrame columns: ['tf', 'target', 'weight'] (optional)
  Set format: {(tf_name, target_gene), ...}
  Values: weight=1 means true edge exists, 0 means no edge
```

---

## 2. GRN Inference Pipeline Architecture

### High-Level Pipeline: Expression Data → Edge Scores → Evaluation

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. LOAD EXPRESSION DATA                                             │
│    DREAMChallengeLoader.load_dream4() / load_dream5_ecoli()         │
│    Returns: (expression, gene_names, tf_names, gold_standard)       │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 2. PREPROCESS DATA                                                  │
│    GRNPreprocessor.fit_transform()                                  │
│    - Normalize expression (zscore/log/quantile/none)                │
│    - Separate TFs (X) from targets (y)                              │
│    - Output: X (n_samples, n_TFs), y (n_samples, n_targets)        │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 3. PREPARE GRN DATA                                                 │
│    GRNDataPipeline.prepare_data()                                   │
│    - Apply train/test split (if requested)                          │
│    - Filter gold standard to relevant edges                         │
│    - Select target subset (max_targets for efficiency)              │
│    - Returns: GRNPreparedData with X_train/y_train etc.            │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 4. TRAIN GRN MODEL (One per Target Gene)                            │
│    TabPFNGRNRegressor.fit(X, y)                                     │
│    For each target_gene in y.columns:                               │
│      - Create TabPFNRegressor(n_estimators=4)                       │
│      - Fit on X (TF expression) → y_target (target gene expr)       │
│      - Store model in target_models_[target_name]                   │
│                                                                      │
│    NO WEIGHT UPDATES: TabPFN is frozen, only in-context learning    │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 5. EXTRACT ATTENTION WEIGHTS (Per Target)                           │
│    AttentionExtractor.extract()                                     │
│    For each fitted model:                                           │
│      - Enable attention weights return on all attention layers      │
│      - Run forward pass (with single-pass mode for both train+test) │
│      - Retrieve and store attention tensors                         │
│      - Returns: dict[layer_key → {between_features, between_items}] │
│                                                                      │
│    Structure: {                                                     │
│      'layer_0': {                                                   │
│        'between_features': (batch, M, M, heads),  # TF-TF attn     │
│        'between_items': (batch, F, F, heads)      # Sample attn    │
│      },                                                             │
│      'layer_1': {...}, ...                                          │
│    }                                                                 │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 6. COMPUTE EDGE SCORES (Extract TF→Target Relationships)            │
│    EdgeScoreComputer.compute() + Strategy-Specific Methods          │
│                                                                      │
│    Strategies:                                                      │
│    - 'self_attention': diag(feat_attn[tf_idx, tf_idx])             │
│    - 'tf_to_target': feat_attn[tf_idx, target_idx]                 │
│    - 'target_to_tf': feat_attn[target_idx, tf_idx]                 │
│    - 'sequential_rollout': Kronecker-factored attention rollout     │
│    - 'gradient_rollout': Gradient-weighted rollout                  │
│    - 'integrated_gradients': IG-based feature attribution           │
│    - 'rise': RISE-based feature attribution                         │
│                                                                      │
│    Output: edge_scores_ = {(tf_name, target_name): score, ...}     │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 7. EVALUATE AGAINST GOLD STANDARD                                   │
│    evaluate_grn() → {auroc, aupr, precision@k, recall@k, f1@k}     │
│                                                                      │
│    Metrics:                                                         │
│    - AUROC: Area Under ROC Curve (false pos rate vs true pos rate)  │
│    - AUPR: Area Under Precision-Recall (primary for sparse networks)│
│    - Precision@k: % of top-k preds that are true                    │
│    - Recall@k: % of true edges recovered in top-k                   │
│    - F1@k: Harmonic mean of precision and recall                    │
└─────────────────────────────────────────────────────────────────────┘
```

### File Locations
- **Datasets loader**: `/src/tabpfn/grn/datasets.py` (DREAMChallengeLoader)
- **Preprocessing**: `/src/tabpfn/grn/preprocessing.py` (GRNPreprocessor)
- **Data pipeline**: `/src/tabpfn/grn/pipeline.py` (GRNDataPipeline, GRNPreparedData)
- **GRN regressor**: `/src/tabpfn/grn/grn_regressor.py` (TabPFNGRNRegressor)
- **Attention extraction**: `/src/tabpfn/grn/attention_extractor.py` (AttentionExtractor, EdgeScoreComputer)
- **Evaluation metrics**: `/src/tabpfn/grn/evaluation.py` (compute_auroc, compute_aupr, evaluate_grn)

---

## 3. GRN Regressor Class (`TabPFNGRNRegressor`)

**Location**: `/src/tabpfn/grn/grn_regressor.py`

### Architecture
```python
class TabPFNGRNRegressor(BaseEstimator):
    def __init__(
        self,
        tf_names: list[str],                          # Feature names
        target_genes: list[str],                      # Target names (one model per target)
        n_estimators: int = 4,                        # Ensemble size
        attention_aggregation: str = "mean",          # How to combine heads/layers
        edge_score_strategy: str = "self_attention",  # Method to extract scores
        device: str = "auto",                         # CPU/CUDA
        use_cross_validation: bool = False,           # CV mode for robust evaluation
        ...
    )
```

### Key Methods

#### `fit(X, y)` → Trained Regressor
- **Input**: `X` (n_samples, n_TFs), `y` (n_samples, n_targets)
- **Process**:
  1. For each target_idx in range(n_targets):
     - Create TabPFNRegressor(n_estimators)
     - Fit on (X, y[:, target_idx])
     - Store in `target_models_[target_name]`
     - Extract attention via AttentionExtractor
     - Store in `attention_weights_[target_name]`
  2. Compute edge scores based on `edge_score_strategy`
  3. Store in `edge_scores_[(tf_name, target_name)]`
  
- **Output**: self (fitted model)

#### `infer_grn(top_k=100)` → NetworkX DiGraph
- Converts edge_scores dict to directed graph
- Returns top-k edges by score
- Format: Nodes are genes, edges indicate TF→target regulation

#### `get_edge_scores()` → dict[(tf, target), score]
- Returns all computed edge scores
- Can be used directly in evaluation metrics

### Fitted Attributes
```python
target_models_: dict[str, TabPFNRegressor]         # One model per target
attention_weights_: dict[str, dict]                # Attention for each target
edge_scores_: dict[tuple[str, str], float]         # (tf, target) → score
X_: np.ndarray                                     # Training data (for gradients)
y_: np.ndarray                                     # Training targets (for gradients)
```

---

## 4. Feature Extraction: How Attention Becomes Edge Scores

### Step 1: Extract Attention from TabPFN

**Class**: `AttentionExtractor` in `/src/tabpfn/grn/attention_extractor.py`

```python
extractor = AttentionExtractor()
attention_weights = extractor.extract(
    model=trained_tabpfn_regressor,
    X=X_test,
    X_train=X_train,  # Optional: for single-pass mode
    y_train=y_train,  # Optional: for single-pass mode
    max_layers=None   # Extract from all layers
)
```

**Returns**:
```python
attention_weights: dict[str, dict[str, torch.Tensor]] = {
    'layer_0': {
        'between_features': (batch, M, M, nheads),  # TF-to-TF attention
        'between_items': (batch, F, F, nheads)      # Sample-to-sample attention
    },
    'layer_1': {...},
    ...
}
```

**Where**:
- `M` = number of samples
- `F` = number of features (TFs + target)
- `nheads` = number of attention heads (typically 6)

### Step 2: Convert Attention to Edge Scores

**Class**: `EdgeScoreComputer` in `/src/tabpfn/grn/attention_extractor.py`

```python
computer = EdgeScoreComputer(aggregation_method="mean")
edge_score_matrix = computer.compute(
    attention_weights,
    use_between_features=True,
    use_between_items=False
)
```

#### Edge Score Strategies Implemented

**1. Self-Attention** (default for small networks)
```
score[tf] = between_features[tf_idx, tf_idx]  # Diagonal of TF attention
# Reasoning: A TF attending to itself suggests it's important for prediction
```

**2. TF-to-Target**
```
score[tf] = between_features[tf_idx, -1]  # Last feature is target gene
# Reasoning: Direct TF→target attention flow in the attention matrix
```

**3. Target-to-TF**
```
score[tf] = between_features[-1, tf_idx]  # Target attending back to TF
# Reasoning: Reciprocal attention indicates tight relationship
```

**4. Sequential Rollout** (Kronecker-factored)
```
Uses Kronecker product to exploit attention structure:
A_feat = feat_attn ⊗ J    (between-features with Kronecker structure)
A_items = J ⊗ item_attn   (between-items with Kronecker structure)

Rollout = A_feat @ A_items @ ... @ A_feat @ A_items
Memory: O(M²F) instead of O(M²F²) via Kronecker decomposition
```

**5. Gradient Rollout** (Advanced)
```
Same as sequential but weights attention heads by gradient importance
weight[head] = ∂L/∂attention[head]  # Gradient of target prediction w.r.t. head
Captures which attention patterns matter most for predicting target gene
```

**6. Integrated Gradients** (State-of-the-art for attribution)
```
Computes feature attribution via gradient interpolation from baseline:
IG[f] = ∫₀¹ ∂f̂(x' + t(x-x'))/∂x[f] dt  where x' = baseline (zero or mean)
Interpretation: Causal importance accumulated from baseline to input
```

**7. RISE** (Randomized Input Sampling)
```
For each TF (feature):
  score[tf] = E_masks[ model(X_masked) - baseline ]
  where masks are random binary masks zeroing out other features
Forward-only, no gradients, avoids gradient coupling issues
```

### Step 3: Aggregate Per-Target Edge Scores

For each target gene, store:
```python
edge_scores[(tf_name, target_name)] = computed_score
```

### Full Feature Extraction Pipeline (Parallel per Target)
```
For target in target_genes:
  1. Fit TabPFNRegressor(X, y_target)
  2. Extract attention_weights via AttentionExtractor
  3. Compute edge scores via EdgeScoreComputer:
     a. Aggregate attention across heads (mean/max) → (M, F, F)
     b. Aggregate across layers (last/mean/max)
     c. Apply strategy (self_attention/tf_to_target/rollout/etc.)
     d. Extract TF-specific scores
  4. Store all (TF, target) → score pairs
```

---

## 5. Evaluation Metrics for GRN Tasks

**Location**: `/src/tabpfn/grn/evaluation.py`

### Primary Metrics (for all predictions)

**AUROC (Area Under ROC Curve)**
```python
compute_auroc(inferred_edges, gold_standard) → float
```
- **Input**: Dict or graph of (tf, target) → score pairs + gold standard edges
- **Output**: Score 0-1 (higher better)
- **Interpretation**: Can the model distinguish true edges from false edges?
- **Formula**: Area under curve of (FPR, TPR)
- **Limitation**: Biased by false negatives in sparse networks

**AUPR (Area Under Precision-Recall Curve)** — RECOMMENDED FOR GRN
```python
compute_aupr(inferred_edges, gold_standard) → float
```
- **Input**: Same as AUROC
- **Output**: Score 0-1 (higher better)
- **Interpretation**: Performance at various confidence thresholds
- **Formula**: Area under curve of (Recall, Precision)
- **Advantage**: More informative for **sparse networks** where true edges << false edges

### Threshold-Based Metrics (for top-k predictions)

**Precision@k**
```python
compute_precision_at_k(inferred_edges, gold_standard, k=100) → float
```
- True positives in top-k / k
- Use case: "How many of my top 100 predictions are correct?"

**Recall@k**
```python
compute_recall_at_k(inferred_edges, gold_standard, k=100) → float
```
- True positives in top-k / total true edges
- Use case: "What fraction of known edges did I recover in top-100?"

**F1@k**
```python
compute_f1_at_k(inferred_edges, gold_standard, k=100) → float
```
- Harmonic mean: 2 × (Precision@k × Recall@k) / (Precision@k + Recall@k)

### Aggregate Evaluation

```python
evaluate_grn(inferred_edges, gold_standard, k_values=[100, 500, 1000])
  → dict with all metrics: AUROC, AUPR, Precision@k, Recall@k, F1@k for each k
```

### Expression Prediction Metrics

For evaluating how well models predict target gene expression:

```python
compute_expression_metrics(y_true, y_pred) → dict
  Returns: MSE, RMSE, MAE, R², Pearson correlation

evaluate_expression_prediction(model, X_train, y_train, X_test, y_test, target_names)
  Returns: Aggregated (mean/std) metrics across all targets
```

---

## 6. Interpretation Model Interface

**Location**: `/src/tabpfn/interpretation/model/interpretation_model.py`

### Architecture Overview

The interpretation model takes **per-feature signal vectors** and predicts **per-feature importance scores**.

### Input: Signal Vectors (from SignalExtractor)

**Shape**: `(batch, n_features, D_total)` where:
- `batch`: Number of samples
- `n_features`: Number of features in the dataset
- `D_total`: Total dimension of per-feature signal vector (concatenation of all internal signals)

**Signal Components** (concatenated into D_total):
```python
signals = {
    'between_features_attention': (n_layers, n_heads) × (n_features, n_features)
    'between_items_attention': (n_layers, n_heads) × (n_samples, n_samples)
    'train_embeddings': (n_train, emsize)  # 192-dim typically
    'test_embeddings': (n_test, emsize)
    'input_gradients': (n_samples, n_features) [optional]
    'attention_gradients': (n_layers, n_heads) × (grad_dim,) [optional]
    'mlp_activations': (n_layers,) × (mlp_hidden_dims)
}
```

Per-feature signal vector (D_total-dimensional):
- Flattened attention patterns for that feature across all layers/heads
- Gradient components for that feature
- MLP activation history for that feature

### Model Variants

#### 1. PerFeatureMLP
```python
class PerFeatureMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,                           # D_total (e.g., 512)
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        output_mode: str = "binary",              # sigmoid or raw
        norm: str = None,                         # layer/batch/None
        activation: str = "relu",
        input_batch_norm: bool = False
    )
    
    def forward(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, n_features, D_total)
        # Flatten to (batch*n_features, D_total)
        # Apply same MLP to each feature's signal vector
        # Reshape back to (batch, n_features)
        # Return: importance scores per feature
```

**Permutation Equivariance**: Same MLP applied independently to each feature makes the model invariant to feature reordering.

#### 2. PerFeatureTransformer
```python
class PerFeatureTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_mode: str = "binary"
    )
    
    def forward(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (batch, n_features, D_total)
        # Project to d_model
        # Apply self-attention ACROSS features (each feature is a token)
        # Output per-feature head
        # Return: (batch, n_features) importance scores
```

**Cross-Feature Attention**: Can model interactions between features (e.g., "feature A is important because feature B is also important").

### Main Interface: InterpretationModel

```python
class InterpretationModel(nn.Module):
    def __init__(
        self,
        variant: str = "mlp",                     # "mlp" or "transformer"
        input_dim: int = 512,                     # D_total
        output_mode: str = "binary",              # "binary" (sigmoid) or "continuous"
        **kwargs                                  # Passed to variant
    )
    
    def forward(
        self,
        x: torch.Tensor,                          # (batch, n_features, D_total)
        mask: torch.Tensor | None = None          # (batch, n_features) validity mask
    ) -> torch.Tensor:
        # Returns: (batch, n_features) importance scores
        return self.model(x, mask=mask)
    
    def predict(self, feature_vectors: torch.Tensor) -> torch.Tensor:
        # feature_vectors: (n_features, D_total)
        # Returns: (n_features,) importance scores
        # Adds batch dim, runs forward, applies sigmoid if binary mode, removes batch
        logits = self.forward(feature_vectors.unsqueeze(0))
        if self.output_mode == "binary":
            return torch.sigmoid(logits).squeeze(0)
        return logits.squeeze(0)
    
    def save(self, path: str | Path) -> None:
        # Saves model checkpoint with variant, input_dim, output_mode
    
    @classmethod
    def from_pretrained(cls, path: str | Path) -> InterpretationModel:
        # Loads model checkpoint
```

### Training Pipeline (from training module)

```python
# See: /src/tabpfn/interpretation/training/trainer.py

from tabpfn.interpretation.training import InterpretationTrainer

trainer = InterpretationTrainer(
    model=InterpretationModel(variant="mlp", input_dim=512),
    learning_rate=0.001,
    epochs=100
)

# Train on (synthetic/real) datasets with ground-truth feature importance labels
metrics = trainer.train(
    train_loader=...,  # Yields (feature_vectors, importance_labels)
    val_loader=...,
    test_loader=...
)

# Model now predicts feature importance from TabPFN signals
```

---

## 7. Existing Benchmark Scripts

**Location**: `/scripts/`

### 1. `grn_performance_analysis.py`
**Purpose**: Comprehensive GRN evaluation across all datasets

**Usage**:
```bash
# Run all datasets
python scripts/grn_performance_analysis.py

# Run specific datasets
python scripts/grn_performance_analysis.py --datasets dream4-10 dream4-100 dream5

# Limit number of networks/targets
python scripts/grn_performance_analysis.py --datasets dream4-10 --max-networks 2 --max-targets 20

# Skip expression prediction
python scripts/grn_performance_analysis.py --no-expression
```

**Functionality**:
- Loads DREAM4 (10, 50, 100) and DREAM5 (E. coli) datasets
- Runs TabPFN GRN inference
- Compares against baselines: GENIE3, GRNBoost2, Mutual Information, Correlation, Linear Regression
- Computes both GRN metrics (AUROC, AUPR) and expression prediction metrics (R², MAE)
- Generates performance report (JSON) and visualizations

**Output**: Results in `results/` directory with:
- `grn_performance_report.json` - All metrics
- `network_visualization.pdf` - Network structure plots
- `score_distribution.pdf` - Edge score histograms
- `pr_curve.pdf` - Precision-Recall curves
- `roc_curve.pdf` - ROC curves

### 2. `test_grn_inference.py`
**Purpose**: Quick validation that GRN pipeline works end-to-end

**Usage**:
```bash
python scripts/test_grn_inference.py
```

**Tests**: 
- Dataset loading
- Preprocessing
- Model training
- Attention extraction
- Edge score computation
- Basic evaluation

### 3. `grn_performance_analysis.py` Functions

Key internal functions for understanding the pipeline:

```python
def evaluate_tabpfn_expression_prediction(model, X_train, y_train, X_test, y_test, ...):
    # Evaluates how well TabPFN predicts target gene expression
    # Returns: {mean_r2, std_r2, mean_mae, ...}

def run_baseline_grn_inference(method, X, y, tf_names, target_genes, gold_standard):
    # Runs one baseline method (GENIE3, GRNBoost2, etc.)
    # Returns: edge_scores dict

def analyze_dataset(
    dataset_name: str,
    expression: np.ndarray,
    gene_names: list[str],
    tf_names: list[str],
    gold_standard: pd.DataFrame,
    max_networks: int | None = None,
    max_targets: int | None = None,
    run_expression: bool = True
):
    # Full analysis pipeline for one dataset
    # Returns: comprehensive metrics dict
```

### 4. `generate_grn_workflow.py`
**Purpose**: Generate workflow YAML for parallelization

### 5. `validate_grn_fix.py`
**Purpose**: Validate specific GRN implementation fixes

---

## 8. Example End-to-End Usage

**Location**: `/examples/grn_inference_example.py`

```python
from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn,
    compute_auroc, compute_aupr
)

# 1. Load Dataset
loader = DREAMChallengeLoader(data_path='data/dream5')
expression, gene_names, tf_names, gold_standard = loader.load_dream4(
    network_size=10, network_id=1
)

# 2. Preprocess
preprocessor = GRNPreprocessor(normalization='zscore')
X, y, tf_indices, target_indices = preprocessor.fit_transform(
    expression, gene_names, tf_names
)
target_genes = preprocessor.get_target_names()

# 3. Train GRN Model
grn = TabPFNGRNRegressor(
    tf_names=tf_names,
    target_genes=target_genes,
    n_estimators=4,
    edge_score_strategy='self_attention'  # or 'tf_to_target', 'rollout', etc.
)
grn.fit(X, y)

# 4. Get Predictions
edge_scores = grn.get_edge_scores()  # dict[(tf, target)] = score
network = grn.infer_grn(top_k=100)   # NetworkX DiGraph

# 5. Evaluate
metrics = evaluate_grn(edge_scores, gold_standard, k_values=[5, 10, 20, 100])
print(f"AUROC: {metrics['auroc']:.4f}")
print(f"AUPR: {metrics['aupr']:.4f}")
print(f"Precision@100: {metrics['precision@100']:.4f}")
```

---

## 9. Key Design Patterns

### 1. Single-Target Model Per Target Gene
- One TabPFNRegressor trained to predict each target gene's expression
- Allows capture of target-specific regulatory patterns
- Enables parallel training (with n_jobs parameter)

### 2. Frozen Foundation Model
- TabPFN's weights are **never** updated
- Only in-context learning is used (examples in forward pass)
- Difference from fine-tuning: no gradient updates to backbone

### 3. Dual Attention Mechanisms
- **Between-features attention**: Captures TF-TF regulatory relationships
- **Between-items attention**: Captures sample-wise patterns
- Both contribute to understanding TF importance

### 4. Kronecker-Factored Rollout
- Exploits tensor structure for memory efficiency: O(M²F) vs O(M²F²)
- Used for sequential_rollout and gradient_rollout strategies
- Mathematically identical to full-matrix rollout but faster

### 5. Multi-Strategy Edge Scoring
- 7 different strategies for extracting edge scores from attention
- Users can choose based on network size and domain:
  - Small networks (10-100 genes): tf_to_target or self_attention
  - Large networks (1000+ genes): gradient_rollout or integrated_gradients

---

## 10. Critical Integration Points

### Connection to Interpretation Model

The **SignalExtractor** bridges GRN inference and interpretation:

```python
from tabpfn.interpretation.extraction import SignalExtractor

# Extract comprehensive signals from TabPFN for interpretation
extractor = SignalExtractor(extract_gradients=True)
signals = extractor.extract(
    model=trained_tabpfn_regressor,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test
)

# Signals include:
# - attention weights (between_features, between_items)
# - embeddings (train + test)
# - gradients (input, attention, embedding)
# - MLP activations

# These signals become input to InterpretationModel:
# feature_vectors = flatten(signals[feature_idx])  # (D_total,)
# importance = interpretation_model.predict(feature_vectors)
```

**Key Difference**:
- **GRN inference** uses **coarse attention patterns** → edge scores
- **Interpretation model** uses **fine-grained internal signals** → feature importance

The interpretation model learns a **non-linear mapping** from signals → labels, potentially capturing richer relationships than attention alone.

---

## 11. Summary Table

| Component | Location | Purpose | Key Classes |
|-----------|----------|---------|-------------|
| **Datasets** | `grn/datasets.py` | Load DREAM benchmark data | DREAMChallengeLoader |
| **Preprocessing** | `grn/preprocessing.py` | Normalize, split TFs/targets | GRNPreprocessor |
| **Pipeline** | `grn/pipeline.py` | Unified data preparation | GRNDataPipeline |
| **GRN Regressor** | `grn/grn_regressor.py` | Main inference model | TabPFNGRNRegressor |
| **Attention Extraction** | `grn/attention_extractor.py` | Extract weights from TabPFN | AttentionExtractor, EdgeScoreComputer |
| **Evaluation** | `grn/evaluation.py` | Metrics (AUROC, AUPR, etc.) | compute_auroc, compute_aupr, evaluate_grn |
| **Visualization** | `grn/visualization.py` | Plots and heatmaps | GRNNetworkVisualizer, EdgeScoreVisualizer |
| **Baseline Methods** | `grn/baseline_models.py` | GENIE3, GRNBoost2, etc. | GRNBaselineRunner |
| **Interpretation Model** | `interpretation/model/interpretation_model.py` | Feature importance prediction | InterpretationModel (PerFeatureMLP, PerFeatureTransformer) |
| **Signal Extraction** | `interpretation/extraction/signal_extractor.py` | Extract TabPFN internal signals | SignalExtractor |
| **Scripts** | `scripts/` | Benchmark running | grn_performance_analysis.py |
| **Examples** | `examples/` | End-to-end usage | grn_inference_example.py |

---

## 12. Performance Results

**DREAM4-10**: TabPFN **WINS** with AUPR=0.75 (vs best baseline 0.18) — **4-5x better!**
**DREAM4-100**: TabPFN competitive with AUPR=0.10
**DREAM5 E.coli**: GENIE3 leads with AUPR=0.22 (TabPFN: 0.16)

TabPFN excels on small synthetic networks where in-context learning shines. On larger real networks, specialized methods maintain advantages.

