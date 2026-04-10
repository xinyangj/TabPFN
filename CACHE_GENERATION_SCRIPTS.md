# TabPFN Interpretation Cache Generation Scripts

## Overview
The TabPFN codebase contains a comprehensive system for generating and managing interpretation cache data. The cache stores feature vectors extracted from TabPFN model's internal signals (attention, embeddings, gradients, etc.) for training interpretation models that predict feature importance.

## Primary Cache Generation Pipeline

### 1. **`scripts/generate_interpretation_data.py`** - Main Cache Generator
**Purpose**: Generate and cache interpretation training data to disk.

**Key Features**:
- Generates synthetic datasets using the external TabPFN prior generator (TabPFNPriorAdapter)
- Runs TabPFN inference to extract internal signals
- Computes ground-truth labels using label generator
- Saves everything to disk as individual .npz files (~1.5 MB per dataset)
- Supports resumable generation and parallel processing across machines

**Output Location**: `data/interpretation_cache/`
**Output Format**: 
- `dataset_XXXXXX.npz` (feature vectors, labels, metadata, raw data)
- `dataset_XXXXXX.json` (metadata as JSON)
- `manifest.json` (summary statistics)

**Key Arguments**:
```bash
--n_datasets 10000        # Number of datasets to generate
--seed 42                 # Random seed for reproducibility
--start_idx 0             # Resume from this index
--cache_dir               # Output directory
--force                   # Regenerate with enriched signals
```

**Example Usage**:
```bash
# Generate 10,000 datasets from scratch
python scripts/generate_interpretation_data.py --n_datasets 10000 --seed 42

# Resume interrupted generation
python scripts/generate_interpretation_data.py --n_datasets 10000 --start_idx 5000
```

**What It Does**:
1. Samples n_features from Beta(0.95, 8.0) distribution [3, 50]
2. Samples n_samples uniformly from [50, 1000]
3. Generates SCM (Structural Causal Model) dataset with TabPFNPriorAdapter
4. Fits TabPFN regressor
5. Extracts 6 signal categories (attention, embeddings, gradients, etc.)
6. Processes signals into per-feature vectors using SignalProcessor
7. Computes binary/ancestry/continuous labels
8. Skips trivial datasets (no direct/ancestry relationships)
9. Saves each dataset individually

---

### 2. **`scripts/augment_gradients.py`** - Add Gradient Features
**Purpose**: Add gradient features to existing cache (backwards-compatible augmentation).

**Key Features**:
- Adds `cat_gradients` to datasets that lack it
- Can load raw data from existing .npz files or regenerate
- Keeps RNG in sync for reproducible regeneration
- Merges gradient vectors into full feature vectors

**Key Arguments**:
```bash
--cache-dir              # Cache directory to augment
--seed 42                # Seed for regeneration (if needed)
--max-datasets 0         # Limit to N datasets (0=all)
```

**Example Usage**:
```bash
python scripts/augment_gradients.py --cache-dir data/interpretation_cache

# Limit to first 1000 datasets for testing
python scripts/augment_gradients.py --cache-dir data/interpretation_cache --max-datasets 1000
```

---

### 3. **`scripts/regenerate_enriched_cache.py`** - Full Signal Re-extraction
**Purpose**: Regenerate cache with enriched signal processing (higher-dimensional features).

**Key Features**:
- Loads raw data (X_train, y_train, X_test) from existing .npz files
- Re-runs TabPFN inference with enriched signal extractor
- Recomputes signals with enriched SignalProcessor (15 stats for between-features attention, 9 for between-items)
- Preserves labels from original files (no re-computation)
- Overwrites in-place or outputs to new directory

**Key Arguments**:
```bash
--start_idx 0            # Resume from this file index
--output_dir             # Output directory (default: overwrite in-place)
--max_files              # Process at most N files
```

**Example Usage**:
```bash
# Regenerate in-place with enriched signals
python scripts/regenerate_enriched_cache.py

# Resume after interruption
python scripts/regenerate_enriched_cache.py --start_idx 5000

# Output to new directory (preserve original)
python scripts/regenerate_enriched_cache.py --output_dir data/interpretation_cache_enriched
```

---

### 4. **`scripts/augment_features_v2.py`** - Enrich Feature Extraction
**Purpose**: Re-extract enriched features from cached raw data using updated SignalProcessor.

**Key Features**:
- Reads raw data and labels from existing cache
- Re-runs TabPFN inference
- Processes with enriched SignalProcessor producing higher-dimensional vectors
- Preserves source cache (writes to separate output directory)
- Resumable and incremental

**Key Arguments**:
```bash
--input-dir              # Source cache directory
--output-dir             # Destination for enriched features
--max-datasets 0         # Limit to N datasets (0=all)
--device                 # GPU device override
```

**Example Usage**:
```bash
python scripts/augment_features_v2.py \
  --input-dir data/interpretation_cache_v2 \
  --output-dir data/interpretation_cache_v2_enriched

# Resume after interruption (skips existing output files)
python scripts/augment_features_v2.py \
  --input-dir data/interpretation_cache_v2 \
  --output-dir data/interpretation_cache_v2_enriched
```

---

## GRN-Specific Cache Scripts

### 5. **`scripts/extract_dream_features.py`** - Extract DREAM4 Benchmark Features
**Purpose**: Extract interpretation features from DREAM4 benchmark datasets for evaluation.

**Key Features**:
- Loads DREAM4 benchmark expression data
- For each target gene, trains TabPFN regressor using TF expression
- Extracts internal signals (attention, activations, gradients)
- Processes into per-feature vectors (691 dims, matching training)
- Caches per-network for fast evaluation

**Output Location**: `data/dream_interpretation_features/`
**Output Format**: `dream4_{size}_net{id}.npz` (per-network feature cache)

**Key Arguments**:
```bash
--device cuda:1          # GPU device
--sizes 10 100           # Network sizes
--networks 1 2 3 4 5     # Network IDs
```

**Example Usage**:
```bash
# Extract features from DREAM4-100 networks
python scripts/extract_dream_features.py --device cuda:1 --sizes 100

# Extract from both 10-gene and 100-gene networks
python scripts/extract_dream_features.py --sizes 10 100 --networks 1 2 3 4 5
```

---

### 6. **`scripts/eval_interpretation_on_grn.py`** - Evaluate on GRN
**Purpose**: Evaluate trained interpretation model on DREAM4 benchmarks.

**Key Features**:
- Loads cached features (from extract_dream_features.py)
- Loads trained interpretation model checkpoint
- Evaluates edge prediction accuracy against gold standard
- Compares to baselines: random, attention heuristic
- Computes AUROC, AUPR, precision@k metrics

**Key Arguments**:
```bash
--device cuda:1          # GPU device
--model                  # Path to trained model checkpoint
--features-dir           # Directory with cached features
--sizes 10 100           # Network sizes to evaluate
```

**Example Usage**:
```bash
python scripts/eval_interpretation_on_grn.py \
  --device cuda:1 \
  --model results/interpretation_experiments/best_v2_model.pt \
  --features-dir data/dream_interpretation_features \
  --sizes 10 100
```

---

## Model Training Scripts

### 7. **`scripts/run_interpretation_experiment.py`** - Training Pipeline
**Purpose**: Comprehensive experiment: train and evaluate interpretation models.

**Key Features**:
- Loads pre-generated datasets from disk (requires generate_interpretation_data.py first)
- Trains interpretation models (MLP and Transformer variants)
- Evaluates across 4 label modes (binary_direct, binary_ancestry, continuous_correlation, continuous_partial_correlation)
- Runs feature ablation study
- Produces detailed summary report

**Requirements**: Pre-generated cache from `generate_interpretation_data.py`

**Example Usage**:
```bash
# First generate data
python scripts/generate_interpretation_data.py --n_datasets 10000

# Then train models
python scripts/run_interpretation_experiment.py
```

---

## Cache Directory Structure

```
data/
├── interpretation_cache/                    # Primary cache (main generation output)
│   ├── dataset_000000.npz                  # Feature vectors, labels, raw data
│   ├── dataset_000000.json                 # Metadata
│   ├── ...
│   └── manifest.json                       # Generation summary
│
├── interpretation_cache_v2/                # Alternative cache version
├── interpretation_cache_v2_enriched/       # Re-extracted with enriched processor
│
└── dream_interpretation_features/           # DREAM4 benchmark features
    ├── dream4_10_net1.npz
    ├── dream4_10_net2.npz
    ├── dream4_100_net1.npz
    └── ...
```

---

## Cache File Format

Each `.npz` file contains:

**Core Data**:
- `feature_vectors`: (n_features, D) feature matrix
- `label_*`: Per-label arrays (binary_direct, binary_ancestry, continuous_*, etc.)

**Raw Data** (for regeneration):
- `raw_X_train`: Training input features
- `raw_y_train`: Training labels
- `raw_X_test`: Test input features

**Per-Category Features**:
- `cat_between_features_attention`: Per-feature attention scores
- `cat_between_items_attention`: Inter-item attention
- `cat_embeddings`: Embedding dimensions
- `cat_mlp_activations`: MLP layer activations
- `cat_gradients`: Gradient-based importance
- `cat_items_attention_gradients`: Gradient of attention weights

**Metadata** (accompanying `.json` file):
- Dataset generation parameters
- SCM metadata
- Feature/sample counts

---

## Typical Cache Generation Workflow

```
1. Generate initial cache
   python scripts/generate_interpretation_data.py --n_datasets 10000

2. (Optional) Augment with gradients
   python scripts/augment_gradients.py --cache-dir data/interpretation_cache

3. Train interpretation model
   python scripts/run_interpretation_experiment.py

4. (Optional) Extract DREAM4 features for GRN evaluation
   python scripts/extract_dream_features.py --device cuda:1

5. Evaluate on GRN benchmarks
   python scripts/eval_interpretation_on_grn.py --device cuda:1
```

---

## Performance Metrics

**Generation Speed**:
- ~1,000-1,500 datasets/hour per GPU (with typical hardware)
- Each dataset: ~50KB-1.5MB depending on n_features

**Cache Versions**:
- Basic: 6 signal categories (generation default)
- Enriched: Higher-dimensional processing (15+ stats per category)

**Evaluation Metrics**:
- AUROC: Area under ROC curve
- AUPR: Area under precision-recall curve  
- Precision@k: Precision at k=number_of_gold_edges
- Early Precision (top 10%): Recovery of edges in top 10%

