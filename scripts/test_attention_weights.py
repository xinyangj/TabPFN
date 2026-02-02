#!/usr/bin/env python
"""Debug script to test attention weight extraction."""

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.grn import DREAMChallengeLoader, GRNPreprocessor
from tabpfn.grn.attention_extractor import AttentionExtractor

# Load a small dataset
loader = DREAMChallengeLoader()
expression, gene_names, tf_names, gold_standard = loader.load_dream4(
    network_size=10, network_id=1
)

# Preprocess
preprocessor = GRNPreprocessor(normalization="zscore")
X, y, tf_indices, target_indices = preprocessor.fit_transform(
    expression, gene_names, tf_names
)
target_genes = preprocessor.get_target_names()

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"TFs: {len(tf_names)}, Targets: {len(target_genes)}")

# Train a simple model for one target
target_idx = 0
target_name = target_genes[target_idx]

print(f"\nTraining for target: {target_name}")

model = TabPFNRegressor(n_estimators=1, device="cpu")
model.fit(X, y[:, target_idx])

# Get the underlying model architecture
print(f"\nChecking model architecture:")
print(f"model.models_: {len(model.models_)} estimators")

model_arch = model.models_[0]
print(f"Has transformer_encoder: {hasattr(model_arch, 'transformer_encoder')}")

if hasattr(model_arch, "transformer_encoder"):
    encoder = model_arch.transformer_encoder
    print(f"Number of layers: {len(encoder.layers)}")

    for layer_idx, layer in enumerate(encoder.layers):
        print(f"\nLayer {layer_idx}:")
        print(f"  Has self_attn_between_features: {hasattr(layer, 'self_attn_between_features')}")
        print(f"  Has self_attn_between_items: {hasattr(layer, 'self_attn_between_items')}")

        if hasattr(layer, "self_attn_between_features") and layer.self_attn_between_features is not None:
            attn_module = layer.self_attn_between_features
            print(f"  attn_module type: {type(attn_module)}")
            print(f"  Has enable_attention_weights_return: {hasattr(attn_module, 'enable_attention_weights_return')}")
            print(f"  Has get_attention_weights: {hasattr(attn_module, 'get_attention_weights')}")

            # Enable attention weights return
            attn_module.enable_attention_weights_return(True)
            print(f"  Enabled attention weights return")

            # Run a forward pass
            print(f"  Running forward pass...")
            with torch.no_grad():
                _ = model.predict(X)

            # Get attention weights
            weights = attn_module.get_attention_weights()
            print(f"  Attention weights: {type(weights)}")
            if weights is not None:
                print(f"    Shape: {weights.shape}")
                print(f"    Min: {weights.min().item():.6f}, Max: {weights.max().item():.6f}")
            else:
                print(f"    None!")

            # Disable
            attn_module.enable_attention_weights_return(False)

print("\n" + "="*70)
print("Debug complete")
