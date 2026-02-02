Add Gene Regulatory Network (GRN) inference module (`tabpfn.grn`)

Features:
- **TabPFNGRNRegressor**: Infer regulatory relationships between transcription factors (TFs) and target genes using TabPFN's attention mechanism
- **In-context learning**: Uses TabPFN as a frozen foundation model (no fine-tuning required)
- **Dual attention extraction**: Captures both between-features (TF-TF) and between-items (sample-wise) attention
- **Standard evaluation metrics**: AUROC, AUPR, Precision@k, Recall@k, F1@k (matching DREAM challenge standards)
- **DREAM challenge support**: Built-in dataset loaders for DREAM3, DREAM4, and DREAM5 benchmarks
- **Comprehensive visualization**: Network graphs, attention heatmaps, precision-recall curves, ROC curves
- **GRNPreprocessor**: Data preprocessing with z-score, log, and quantile normalization
- **EdgeScoreComputer**: Aggregate attention weights across layers/heads/samples

Example:
```python
from tabpfn.grn import DREAMChallengeLoader, GRNPreprocessor, TabPFNGRNRegressor, evaluate_grn

# Load data
loader = DREAMChallengeLoader()
expression, gene_names, tf_names, gold_standard = loader.load_dream4(network_size=10)

# Preprocess
preprocessor = GRNPreprocessor(normalization="zscore")
X, y, tf_indices, target_indices = preprocessor.fit_transform(expression, gene_names, tf_names)

# Train
grn = TabPFNGRNRegressor(tf_names, preprocessor.get_target_names(), n_estimators=1)
grn.fit(X, y)

# Evaluate
metrics = evaluate_grn(grn.get_edge_scores(), gold_standard)
print(f"AUPR: {metrics['aupr']:.4f}, AUROC: {metrics['auroc']:.4f}")
```

Documentation:
- Tutorial: `examples/notebooks/GRN_Inference_Tutorial.ipynb`
- Example: `examples/grn_inference_example.py`
- Plan: `GRN_IMPLEMENTATION_PLAN.md`
