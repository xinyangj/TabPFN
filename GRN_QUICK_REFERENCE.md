# GRN Infrastructure - Quick Reference

## For Connecting Interpretation Model to GRN Inference

### What You Need to Know

**1. GRN Inference Output**: `TabPFNGRNRegressor`
- Trains one `TabPFNRegressor` per target gene (target_genes list)
- Extracts attention weights via `AttentionExtractor`
- Converts attention to edge scores: `(tf_name, target_name) → score`
- **Key insight**: Edge scores come from attention patterns, not gradients

**2. Interpretation Model Input**: Per-feature signal vectors
- Shape: `(n_features, D_total)` 
- Composed of: flattened attention, embeddings, gradients, MLP activations
- Produced by: `SignalExtractor` from `interpretation/extraction/`
- **Key insight**: Rich internal signals → learns implicit feature importance mapping

**3. The Bridge - SignalExtractor**
```python
from tabpfn.interpretation.extraction import SignalExtractor

# Extract all internal signals from one TabPFN prediction
extractor = SignalExtractor(extract_gradients=True)
signals = extractor.extract(model, X_train, y_train, X_test)
# Returns: attention, embeddings, gradients, MLP activations

# These signals become input to InterpretationModel
# Unlike GRN edge scores, interpretation model sees COMPLETE internal state
```

### Two Parallel Feature Importance Approaches

| Aspect | GRN Edge Scores | Interpretation Model |
|--------|-----------------|---------------------|
| **How** | Extract from attention patterns | Learn from internal signals |
| **Input** | between_features attention matrix | Comprehensive signal vector |
| **Output** | Edge score per (TF, target) pair | Feature importance per feature |
| **Interpretability** | Direct attention routing | Learned non-linear mapping |
| **Applicable to** | Regulatory relationships | General feature importance |
| **Speed** | Fast (single forward pass) | Fast (single forward pass) |

### Potential Integration

**Option A: Fusion Model**
- Run both GRN and interpretation model on same data
- Combine: `edge_score + interpretation_importance` 
- Might capture complementary signals

**Option B: Interpretation Model as Validation**
- GRN produces edge scores (coarse)
- Interpretation model validates which TFs are "truly important" (fine-grained)
- Rank edges by: `score × importance_weight`

**Option C: Feature Extraction from GRN for Interpretation**
- Use attention patterns from GRN as input features to interpretation model
- Interpretation model learns: "which attention patterns predict true edges?"

---

## Key File Locations

```
GRN INFERENCE:
├── src/tabpfn/grn/
│   ├── datasets.py              # DREAMChallengeLoader (data loading)
│   ├── preprocessing.py         # GRNPreprocessor (normalize, split)
│   ├── pipeline.py              # GRNDataPipeline (unified data prep)
│   ├── grn_regressor.py         # TabPFNGRNRegressor (main model)
│   ├── attention_extractor.py   # AttentionExtractor, EdgeScoreComputer
│   ├── evaluation.py            # compute_auroc, compute_aupr, evaluate_grn
│   └── baseline_models.py       # Baseline methods (GENIE3, etc.)

INTERPRETATION MODEL:
├── src/tabpfn/interpretation/
│   ├── model/
│   │   ├── interpretation_model.py   # PerFeatureMLP, PerFeatureTransformer, InterpretationModel
│   │   └── losses.py                 # Training losses
│   ├── extraction/
│   │   ├── signal_extractor.py       # SignalExtractor (tabpfn→signals)
│   │   └── signal_processor.py       # Signal processing utilities
│   ├── training/
│   │   ├── trainer.py                # InterpretationTrainer
│   │   ├── dataset.py                # Training dataset loaders
│   │   └── data_pipeline.py          # Data preparation
│   ├── synthetic_data/               # Synthetic benchmark generation
│   └── evaluation/
│       ├── benchmarks.py             # Evaluation suites
│       └── metrics.py                # Metric functions

BENCHMARK DATA:
├── data/
│   ├── dream4/dream4/              # DREAM4 networks (10, 50, 100 genes)
│   │   └── dream4_{size}_net{id}_*
│   └── dream5/                      # DREAM5 networks (E. coli, yeast, etc.)

SCRIPTS:
├── scripts/
│   ├── grn_performance_analysis.py  # Main benchmark runner
│   ├── test_grn_inference.py        # Quick validation
│   └── ...

EXAMPLES:
├── examples/
│   └── grn_inference_example.py     # End-to-end GRN usage
```

---

## Critical Code Patterns

### 1. Running Full GRN Pipeline
```python
from tabpfn.grn import (
    DREAMChallengeLoader,
    GRNPreprocessor,
    TabPFNGRNRegressor,
    evaluate_grn
)

# Load
loader = DREAMChallengeLoader(data_path='data/dream4')
X_raw, genes, tfs, gold = loader.load_dream4(network_size=10, network_id=1)

# Preprocess
prep = GRNPreprocessor(normalization='zscore')
X, y, _, _ = prep.fit_transform(X_raw, genes, tfs)
targets = prep.get_target_names()

# Train
grn = TabPFNGRNRegressor(tf_names=tfs, target_genes=targets)
grn.fit(X, y)

# Evaluate
edge_scores = grn.get_edge_scores()
metrics = evaluate_grn(edge_scores, gold)
print(f"AUPR: {metrics['aupr']:.4f}")
```

### 2. Extracting Signals for Interpretation Model
```python
from tabpfn.interpretation.extraction import SignalExtractor
from tabpfn.interpretation.model import InterpretationModel

# Train a TabPFN regressor
regressor = TabPFNRegressor(n_estimators=1)
regressor.fit(X_train, y_train)

# Extract all internal signals
extractor = SignalExtractor(extract_gradients=True)
signals = extractor.extract(regressor, X_train, y_train, X_test)

# Prepare per-feature signal vectors (shape: n_features × D_total)
feature_signals = prepare_feature_signals(signals)  # Custom function

# Predict feature importance
interp_model = InterpretationModel.from_pretrained('path/to/model.pt')
importances = interp_model.predict(feature_signals)
```

### 3. Understanding Attention Extraction
```python
from tabpfn.grn.attention_extractor import AttentionExtractor

extractor = AttentionExtractor()

# Single-pass extraction (captures both train AND predict phases)
attention = extractor.extract(
    model=fitted_tabpfn_regressor,
    X=X_combined,          # Combined train+test
    X_train=X_train,       # Just for info
    y_train=y_train,       # Training labels (auto sets single_eval_pos)
    max_layers=None        # All layers
)

# Structure:
# attention['layer_0'] = {
#     'between_features': (batch, n_samples, n_samples, n_heads),
#     'between_items': (batch, n_features, n_features, n_heads)
# }
```

---

## Edge Score Strategies Explained

| Strategy | Formula | Best For | Computation |
|----------|---------|----------|-------------|
| `self_attention` | `attn[tf_idx, tf_idx]` | Small networks, TF importance | O(n_tfs × n_layers) |
| `tf_to_target` | `attn[tf_idx, target_idx]` | Direct TF→target flow | O(n_tfs × n_layers) |
| `target_to_tf` | `attn[target_idx, tf_idx]` | Reciprocal relationships | O(n_tfs × n_layers) |
| `sequential_rollout` | `rollout = ∏(A_feat @ A_items)` | Large networks, fine-grained | O(M²F) via Kronecker |
| `gradient_rollout` | `rollout weighted by ∂L/∂head` | When gradients matter | O(M²F) + gradient pass |
| `integrated_gradients` | `∫∂f̂/∂x from baseline` | Causal attribution | Expensive (multiple passes) |
| `rise` | `E_masks[model(X_masked) - baseline]` | Forward-only, no gradients | Multiple random passes |

---

## Metrics Understanding

**AUROC vs AUPR**
- **AUROC**: Good for balanced datasets
- **AUPR**: Better for sparse GRNs (true edges << false edges)
  - Why? Precision-recall emphasizes rare positives
  - DREAM4-10: AUPR is primary metric
  - DREAM5 (real): Still use AUPR despite imbalance

**Precision@k, Recall@k, F1@k**
- Used to evaluate top-k predictions
- Typical k values: 100, 500, 1000
- Experimental validation budget: top-100 predictions

---

## Performance Benchmarks (Current State)

| Dataset | TabPFN AUPR | Best Baseline | Note |
|---------|-------------|---------------|------|
| DREAM4-10 | **0.75** ✓ | 0.18 (MI) | Huge win! |
| DREAM4-50 | 0.20 | 0.19 (Corr) | Competitive |
| DREAM4-100 | 0.10 | 0.10 (Corr) | Even match |
| DREAM5 E.coli | 0.16 | 0.22 (GENIE3) | GENIE3 ahead |

**Insight**: TabPFN excels on small synthetic networks where in-context learning is most powerful. On larger/real networks, specialized methods maintain advantages.

---

## Next Steps for Integration

1. **Run GRN on sample data**: `python examples/grn_inference_example.py`
2. **Understand signal extraction**: Check `SignalExtractor.extract()` output structure
3. **Train interpretation model**: Use `/src/tabpfn/interpretation/training/trainer.py`
4. **Design fusion strategy**: Combine edge scores + importance weights
5. **Benchmark on DREAM4-10**: Should see TabPFN's strength

