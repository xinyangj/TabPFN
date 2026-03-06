"""Process raw TabPFN signals into fixed-size per-feature vectors.

Transforms variable-size attention matrices, embeddings, gradients,
and activations into a fixed-dimensional feature vector for each input
feature, suitable for the interpretation model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class SignalProcessor:
    """Process extracted TabPFN signals into per-feature feature vectors.

    Takes the raw signals from SignalExtractor and computes per-feature
    statistics that form the input to the interpretation model.

    The output is a matrix of shape (n_features, D_total) where D_total
    is the total dimension of all processed signal categories.

    Examples
    --------
    >>> processor = SignalProcessor()
    >>> feature_vectors = processor.process(signals)
    >>> print(feature_vectors.shape)  # (n_features, D_total)
    """

    def process(self, signals: dict[str, Any]) -> np.ndarray:
        """Process all signals into per-feature feature vectors.

        Parameters
        ----------
        signals : dict
            Output from SignalExtractor.extract().

        Returns
        -------
        np.ndarray
            Per-feature feature matrix of shape (n_features, D_total).
        """
        n_features = signals["n_features"]

        feature_parts = []

        # 1. Between-features attention features
        feat_attn = self._process_between_features_attention(
            signals["between_features_attention"], n_features
        )
        if feat_attn is not None:
            feature_parts.append(feat_attn)

        # 2. Between-items attention features
        item_attn = self._process_between_items_attention(
            signals["between_items_attention"], n_features
        )
        if item_attn is not None:
            feature_parts.append(item_attn)

        # 3. Embedding features
        emb_feats = self._process_embeddings(signals, n_features)
        if emb_feats is not None:
            feature_parts.append(emb_feats)

        # 4. Gradient features
        grad_feats = self._process_gradients(signals, n_features)
        if grad_feats is not None:
            feature_parts.append(grad_feats)

        # 5. MLP activation features
        act_feats = self._process_mlp_activations(
            signals["mlp_activations"], n_features
        )
        if act_feats is not None:
            feature_parts.append(act_feats)

        if not feature_parts:
            raise ValueError("No signals could be processed.")

        return np.concatenate(feature_parts, axis=1)

    def _process_between_features_attention(
        self,
        attention_dict: dict[str, torch.Tensor | None],
        n_features: int,
    ) -> np.ndarray | None:
        """Process between-features attention into per-feature vectors.

        For each feature i, extracts:
        - Self-attention: attn[i, i] per layer per head
        - Attention to target: attn[i, -1] per layer per head
        - Attention from target: attn[-1, i] per layer per head
        - Mean attention to/from other features
        - Attention entropy
        """
        # Collect valid attention tensors
        layer_attns = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None and not torch.isnan(attn).any():
                layer_attns.append(attn.cpu().float())

        if not layer_attns:
            return None

        n_layers = len(layer_attns)
        # Get feature block mapping
        # Attention shape: (batch_items, n_heads, n_blocks, n_blocks)
        # where n_blocks = ceil(n_features * 2 / features_per_group) + 1
        # Last block is the target
        n_heads = layer_attns[0].shape[-3]

        features_per_block = []
        for attn in layer_attns:
            n_blocks = attn.shape[-1]
            features_per_block.append(n_blocks)

        n_blocks = features_per_block[0]
        target_block = n_blocks - 1

        # Map feature indices to block indices (simplified mapping)
        # Each feature gets ~1 block (with features_per_group=2, each feature
        # produces 2 encoded values, and 2 encoded values = 1 block)
        feature_block_map = self._compute_feature_block_mapping(n_features, n_blocks)

        per_feature_vectors = []
        for feat_idx in range(n_features):
            block_idx = feature_block_map[feat_idx]
            feat_vector = []

            for layer_idx, attn in enumerate(layer_attns):
                # Average over batch dimension
                attn_avg = attn.mean(dim=0)  # (n_heads, n_blocks, n_blocks)

                for head_idx in range(n_heads):
                    a = attn_avg[head_idx]  # (n_blocks, n_blocks)

                    # Self-attention
                    feat_vector.append(float(a[block_idx, block_idx]))

                    # Attention to target
                    feat_vector.append(float(a[block_idx, target_block]))

                    # Attention from target
                    feat_vector.append(float(a[target_block, block_idx]))

                    # Mean attention to other features
                    other_mask = torch.ones(n_blocks, dtype=torch.bool)
                    other_mask[block_idx] = False
                    other_mask[target_block] = False
                    if other_mask.any():
                        feat_vector.append(float(a[block_idx, other_mask].mean()))
                        feat_vector.append(float(a[other_mask, block_idx].mean()))
                    else:
                        feat_vector.append(0.0)
                        feat_vector.append(0.0)

                    # Attention entropy
                    row = a[block_idx]
                    row_clamped = row.clamp(min=1e-10)
                    entropy = -float((row_clamped * row_clamped.log()).sum())
                    feat_vector.append(entropy)

            # Cross-layer statistics for attention-to-target
            attn_to_target_per_layer = []
            for layer_idx, attn in enumerate(layer_attns):
                attn_avg = attn.mean(dim=0)
                vals = [float(attn_avg[h, block_idx, target_block]) for h in range(n_heads)]
                attn_to_target_per_layer.append(np.mean(vals))

            feat_vector.append(np.mean(attn_to_target_per_layer))
            feat_vector.append(np.std(attn_to_target_per_layer))
            if len(attn_to_target_per_layer) > 1:
                # Linear trend
                x_layer = np.arange(len(attn_to_target_per_layer))
                trend = np.polyfit(x_layer, attn_to_target_per_layer, 1)[0]
                feat_vector.append(trend)
            else:
                feat_vector.append(0.0)

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    def _process_between_items_attention(
        self,
        attention_dict: dict[str, torch.Tensor | None],
        n_features: int,
    ) -> np.ndarray | None:
        """Process between-items attention into per-feature summary statistics.

        Extracts per-feature: entropy, concentration, train-test attention ratio.
        """
        layer_attns = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None and not torch.isnan(attn).any():
                layer_attns.append(attn.cpu().float())

        if not layer_attns:
            return None

        n_layers = len(layer_attns)
        # Shape: (batch * n_feat_blocks, n_heads, n_items, n_items)
        n_heads = layer_attns[0].shape[-3]

        # We compute summary statistics per feature block
        n_blocks_per_layer = []
        for attn in layer_attns:
            # Items attention is transposed: (batch * n_feat_blocks, n_heads, n_items, n_items)
            total_batch = attn.shape[0]
            n_blocks_per_layer.append(total_batch)

        per_feature_vectors = []
        for feat_idx in range(n_features):
            feat_vector = []

            for layer_idx, attn in enumerate(layer_attns):
                n_total = attn.shape[0]
                # Estimate batch_size and n_feat_blocks
                # For a single batch item, n_total = n_feat_blocks
                # We take the feat_idx-th block (clamped to available blocks)
                block_idx = min(feat_idx, n_total - 1)

                for head_idx in range(n_heads):
                    a = attn[block_idx, head_idx]  # (n_items, n_items)

                    # Entropy of attention distribution (mean across items)
                    row_entropies = []
                    for row in a:
                        row_clamped = row.clamp(min=1e-10)
                        ent = -float((row_clamped * row_clamped.log()).sum())
                        row_entropies.append(ent)
                    feat_vector.append(np.mean(row_entropies))

                    # Max attention (concentration)
                    feat_vector.append(float(a.max()))

                    # Attention variance
                    feat_vector.append(float(a.var()))

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    def _process_embeddings(
        self, signals: dict[str, Any], n_features: int
    ) -> np.ndarray | None:
        """Process encoder embeddings into per-feature vectors."""
        train_emb = signals.get("train_embeddings")
        test_emb = signals.get("test_embeddings")

        if train_emb is None and test_emb is None:
            return None

        per_feature_vectors = []
        for feat_idx in range(n_features):
            feat_vector = []

            if test_emb is not None:
                emb = test_emb.cpu().float()
                # Embeddings shape: (n_test, batch, emsize) or (n_test, emsize)
                if emb.ndim == 3:
                    emb = emb[:, 0, :]  # Take first batch

                # Per-feature: need to identify which embedding dim corresponds to feature
                # For now use the full embedding as a global context signal
                # Mean, std, max across samples
                feat_vector.extend(emb.mean(dim=0).numpy().tolist())
                feat_vector.extend(emb.std(dim=0).numpy().tolist())

            if train_emb is not None:
                emb = train_emb.cpu().float()
                if emb.ndim == 3:
                    emb = emb[:, 0, :]
                feat_vector.extend(emb.mean(dim=0).numpy().tolist())

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    def _process_gradients(
        self, signals: dict[str, Any], n_features: int
    ) -> np.ndarray | None:
        """Process gradients into per-feature vectors."""
        input_grads = signals.get("input_gradients")
        attn_grads = signals.get("attention_gradients")

        if input_grads is None and attn_grads is None:
            return None

        per_feature_vectors = []
        for feat_idx in range(n_features):
            feat_vector = []

            # Input gradients
            if input_grads is not None:
                grads = input_grads.cpu().float()
                if grads.ndim == 3:
                    grads = grads[:, 0, :]  # Remove batch dim
                if feat_idx < grads.shape[-1]:
                    feat_grads = grads[:, feat_idx]
                    feat_vector.append(float(feat_grads.abs().mean()))
                    feat_vector.append(float(feat_grads.abs().max()))
                    feat_vector.append(float(feat_grads.std()))
                    # Sign consistency
                    pos_frac = float((feat_grads > 0).float().mean())
                    feat_vector.append(max(pos_frac, 1.0 - pos_frac))
                else:
                    feat_vector.extend([0.0, 0.0, 0.0, 0.5])

            # Attention gradients
            if attn_grads is not None:
                for key in sorted(attn_grads.keys()):
                    grad = attn_grads[key]
                    if grad is not None:
                        g = grad.cpu().float()
                        # Average over batch dimension
                        g_avg = g.mean(dim=0)  # (n_heads, n_blocks, n_blocks)
                        n_heads = g_avg.shape[0]
                        n_blocks = g_avg.shape[-1]
                        block_idx = min(feat_idx, n_blocks - 2)
                        target_block = n_blocks - 1

                        for h in range(n_heads):
                            # Gradient of attention from feature to target
                            feat_vector.append(float(g_avg[h, block_idx, target_block]))
                            # Gradient magnitude mean
                            feat_vector.append(float(g_avg[h, block_idx].abs().mean()))
                    else:
                        # Placeholder zeros for missing layers
                        feat_vector.extend([0.0, 0.0] * 6)  # 6 heads

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    def _process_mlp_activations(
        self,
        activation_dict: dict[int, torch.Tensor],
        n_features: int,
    ) -> np.ndarray | None:
        """Process MLP activations into per-feature vectors."""
        if not activation_dict:
            return None

        per_feature_vectors = []
        for feat_idx in range(n_features):
            feat_vector = []

            for layer_idx in sorted(activation_dict.keys()):
                act = activation_dict[layer_idx].cpu().float()
                # Act shape: (seq_len, batch*n_feat_blocks, hidden_dim) or similar
                # Take mean over all dimensions except the hidden dim
                if act.ndim >= 2:
                    act_flat = act.reshape(-1, act.shape[-1])
                    # Per-feature block statistics
                    block_idx = min(feat_idx, act_flat.shape[0] - 1)
                    act_block = act_flat[block_idx]

                    feat_vector.append(float(act_block.mean()))
                    feat_vector.append(float(act_block.std()))
                    feat_vector.append(float(act_block.max()))
                    # Sparsity (fraction near zero)
                    feat_vector.append(float((act_block.abs() < 0.01).float().mean()))
                    feat_vector.append(float(act_block.norm()))

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    @staticmethod
    def _compute_feature_block_mapping(
        n_features: int, n_blocks: int
    ) -> dict[int, int]:
        """Map raw feature indices to attention block indices.

        TabPFN encodes each feature as 2 values (value + NaN indicator),
        and groups every features_per_group=2 encoded values into 1 block.
        The last block is the target.
        """
        # n_blocks = n_feature_blocks + 1 (target block)
        n_feature_blocks = n_blocks - 1
        mapping = {}
        for i in range(n_features):
            # Each feature maps to approximately 1 block
            block = min(i, n_feature_blocks - 1)
            mapping[i] = block
        return mapping
