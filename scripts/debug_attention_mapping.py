#!/usr/bin/env python
"""Debug script to understand attention-to-TF mapping."""

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.grn import DREAMChallengeLoader, GRNPreprocessor

# Load dataset
loader = DREAMChallengeLoader()
expression, gene_names, tf_names, gold_standard = loader.load_dream4(
    network_size=10, network_id=1
)

preprocessor = GRNPreprocessor(normalization="zscore")
X, y, tf_indices, target_indices = preprocessor.fit_transform(
    expression, gene_names, tf_names
)
target_genes = preprocessor.get_target_names()

print(f"Input X shape: {X.shape}")
print(f"TF names: {tf_names}")
print(f"Target genes: {target_genes}")

# Train model for one target
target_idx = 0
target_name = target_genes[target_idx]
print(f"\nTraining for target: {target_name}")

model = TabPFNRegressor(n_estimators=1, device="cpu")
model.fit(X, y[:, target_idx])

# Enable attention weights return and extract
model_arch = model.models_[0]
encoder = model_arch.transformer_encoder
layer = encoder.layers[0]
attn_module = layer.self_attn_between_features

attn_module.enable_attention_weights_return(True)
with torch.no_grad():
    _ = model.predict(X)
attention = attn_module.get_attention_weights()
attn_module.enable_attention_weights_return(False)

print(f"\nAttention shape: {attention.shape}")
print(f"  Expected: [seq_len, n_feat_pos, n_feat_pos, n_heads]")

# Analyze the attention
# Aggregate across samples and heads
feat_attn = attention.mean(dim=0).mean(dim=-1)  # [n_feat_pos, n_feat_pos]
print(f"\nFeature-feature attention: {feat_attn.shape}")
print(feat_attn)

# Check each TF position
print(f"\nAttention from each TF position:")
for i in range(feat_attn.shape[0]):
    print(f"  Position {i}: {feat_attn[i, :].tolist()}")

# Try to understand the mapping
print(f"\nTrying different aggregation strategies:")
print(f"  Row means: {feat_attn.mean(dim=1)}")
print(f"  Column means: {feat_attn.mean(dim=0)}")
print(f"  Diagonal: {torch.diag(feat_attn)}")
print(f"  Upper triangle mean: {torch.triu(feat_attn).mean()}")
print(f"  Lower triangle mean: {torch.tril(feat_attn).mean()}")

# Try to use the attention weights directly as TF scores
# Assuming the first n_tfs feature positions correspond to TFs
print(f"\nDirect attention scores for each TF:")
for tf_idx, tf_name in enumerate(tf_names):
    if tf_idx < feat_attn.shape[0]:
        # Use the average attention from this position to all others
        score1 = feat_attn[tf_idx, :].mean().item()
        # Use the average attention from all others to this position
        score2 = feat_attn[:, tf_idx].mean().item()
        # Use the diagonal
        score3 = feat_attn[tf_idx, tf_idx].item()
        print(f"  {tf_name}: row_mean={score1:.4f}, col_mean={score2:.4f}, diag={score3:.4f}")
