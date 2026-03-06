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

        result = np.concatenate(feature_parts, axis=1)
        # Clean up any NaN/Inf values
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _process_between_features_attention(
        self,
        attention_dict: dict[str, torch.Tensor | None],
        n_features: int,
    ) -> np.ndarray | None:
        """Process between-features attention into per-feature vectors.

        Attention shape from TabPFN: (batch_items, n_blocks, n_blocks, n_heads)
        where n_blocks = ceil(n_features / features_per_group) + 1 (last = target).

        For each feature i, extracts per layer per head:
        - Self-attention, attention to/from target, mean attention to/from others, entropy
        Plus cross-layer statistics.
        """
        layer_attns = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None:
                a = attn.cpu().float()
                a = torch.nan_to_num(a, nan=0.0)
                layer_attns.append(a)

        if not layer_attns:
            return None

        # Shape: (batch_items, n_blocks, n_blocks, n_heads)
        n_blocks = layer_attns[0].shape[1]
        n_heads = layer_attns[0].shape[3]
        target_block = n_blocks - 1
        n_feature_blocks = n_blocks - 1

        # Map features to blocks: features_per_group features share one block
        feature_block_map = {}
        for i in range(n_features):
            feature_block_map[i] = min(i * n_feature_blocks // n_features, n_feature_blocks - 1)

        per_feature_vectors = []
        for feat_idx in range(n_features):
            block_idx = feature_block_map[feat_idx]
            feat_vector = []

            for attn in layer_attns:
                # Average over batch_items dimension: (n_blocks, n_blocks, n_heads)
                a = attn.mean(dim=0)

                for h in range(n_heads):
                    ah = a[:, :, h]  # (n_blocks, n_blocks) for this head

                    # Self-attention
                    feat_vector.append(float(ah[block_idx, block_idx]))
                    # Attention to target
                    feat_vector.append(float(ah[block_idx, target_block]))
                    # Attention from target
                    feat_vector.append(float(ah[target_block, block_idx]))

                    # Mean attention to/from other feature blocks
                    other_blocks = [b for b in range(n_feature_blocks) if b != block_idx]
                    if other_blocks:
                        feat_vector.append(float(ah[block_idx, other_blocks].mean()))
                        feat_vector.append(float(ah[other_blocks, :][:, block_idx].mean()))
                    else:
                        feat_vector.extend([0.0, 0.0])

                    # Attention entropy
                    row = ah[block_idx].clamp(min=1e-10)
                    entropy = -float((row * row.log()).sum())
                    feat_vector.append(entropy)

            # Cross-layer statistics for attention-to-target
            attn_to_target_per_layer = []
            for attn in layer_attns:
                a = attn.mean(dim=0)  # (n_blocks, n_blocks, n_heads)
                vals = [float(a[block_idx, target_block, h]) for h in range(n_heads)]
                attn_to_target_per_layer.append(np.mean(vals))

            feat_vector.append(np.mean(attn_to_target_per_layer))
            feat_vector.append(np.std(attn_to_target_per_layer))
            if len(attn_to_target_per_layer) > 1:
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

        Attention shape: (n_blocks, n_items, n_items, n_heads)
        Extracts per-feature: entropy, concentration, attention variance.
        """
        layer_attns = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None:
                a = attn.cpu().float()
                a = torch.nan_to_num(a, nan=0.0)
                layer_attns.append(a)

        if not layer_attns:
            return None

        # Shape: (n_blocks, n_items, n_items, n_heads)
        n_blocks_attn = layer_attns[0].shape[0]
        n_heads = layer_attns[0].shape[3]

        # Map features to blocks
        n_feature_blocks = n_blocks_attn
        feature_block_map = {}
        for i in range(n_features):
            feature_block_map[i] = min(i * n_feature_blocks // max(n_features, 1), n_feature_blocks - 1)

        per_feature_vectors = []
        for feat_idx in range(n_features):
            block_idx = feature_block_map[feat_idx]
            feat_vector = []

            for attn in layer_attns:
                # attn shape: (n_blocks, n_items, n_items, n_heads)
                a_block = attn[min(block_idx, attn.shape[0] - 1)]  # (n_items, n_items, n_heads)

                for h in range(n_heads):
                    ah = a_block[:, :, h]  # (n_items, n_items)

                    # Mean entropy across items
                    row_entropies = []
                    for row in ah:
                        row_c = row.clamp(min=1e-10)
                        ent = -float((row_c * row_c.log()).sum())
                        row_entropies.append(ent)
                    feat_vector.append(np.mean(row_entropies))

                    # Max attention (concentration)
                    feat_vector.append(float(ah.max()))

                    # Attention variance
                    feat_vector.append(float(ah.var()))

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    def _process_embeddings(
        self, signals: dict[str, Any], n_features: int
    ) -> np.ndarray | None:
        """Process encoder embeddings into per-feature vectors.

        Embeddings shape: (n_samples, batch=1, emsize=192).
        These are global (not per-feature), so we use them as shared context.
        """
        train_emb = signals.get("train_embeddings")
        test_emb = signals.get("test_embeddings")

        if train_emb is None and test_emb is None:
            return None

        # Compute global embedding statistics
        global_vector = []
        if test_emb is not None:
            emb = test_emb.cpu().float()
            if emb.ndim == 3:
                emb = emb[:, 0, :]  # (n_test, emsize)
            global_vector.extend(emb.mean(dim=0).numpy().tolist())
            global_vector.extend(emb.std(dim=0).numpy().tolist())

        if train_emb is not None:
            emb = train_emb.cpu().float()
            if emb.ndim == 3:
                emb = emb[:, 0, :]
            global_vector.extend(emb.mean(dim=0).numpy().tolist())

        # Replicate global vector for each feature
        global_arr = np.array(global_vector, dtype=np.float32)
        return np.tile(global_arr, (n_features, 1))

    def _process_gradients(
        self, signals: dict[str, Any], n_features: int
    ) -> np.ndarray | None:
        """Process gradients into per-feature vectors.

        Input gradient shape: (n_samples, batch=1, n_features).
        Attention gradient shape: (batch_items, n_blocks, n_blocks, n_heads).
        """
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
                # Shape: (seq_len, batch=1, n_features)
                if grads.ndim == 3:
                    grads = grads[:, 0, :]  # (seq_len, n_features)
                if feat_idx < grads.shape[-1]:
                    feat_grads = grads[:, feat_idx]
                    feat_vector.append(float(feat_grads.abs().mean()))
                    feat_vector.append(float(feat_grads.abs().max()))
                    feat_vector.append(float(feat_grads.std()))
                    pos_frac = float((feat_grads > 0).float().mean())
                    feat_vector.append(max(pos_frac, 1.0 - pos_frac))
                else:
                    feat_vector.extend([0.0, 0.0, 0.0, 0.5])

            # Attention gradients: (batch_items, n_blocks, n_blocks, n_heads)
            if attn_grads is not None:
                for key in sorted(attn_grads.keys()):
                    grad = attn_grads[key]
                    if grad is not None:
                        g = grad.cpu().float()
                        g_avg = g.mean(dim=0)  # (n_blocks, n_blocks, n_heads)
                        n_blocks = g_avg.shape[0]
                        n_heads = g_avg.shape[2]
                        n_feat_blocks = n_blocks - 1
                        block_idx = min(feat_idx * n_feat_blocks // max(n_features, 1), n_feat_blocks - 1)
                        target_block = n_blocks - 1

                        for h in range(n_heads):
                            feat_vector.append(float(g_avg[block_idx, target_block, h]))
                            feat_vector.append(float(g_avg[block_idx, :, h].abs().mean()))
                    else:
                        feat_vector.extend([0.0, 0.0] * 3)

            per_feature_vectors.append(feat_vector)

        return np.array(per_feature_vectors, dtype=np.float32)

    def _process_mlp_activations(
        self,
        activation_dict: dict[int, torch.Tensor],
        n_features: int,
    ) -> np.ndarray | None:
        """Process MLP activations into per-feature vectors.

        Activation shape: (batch, batch_items, n_blocks, emsize=192).
        We extract per-block statistics averaged over items.
        """
        if not activation_dict:
            return None

        # Determine block mapping
        first_act = list(activation_dict.values())[0]
        # Shape: (batch=1, batch_items, n_blocks, emsize)
        n_blocks = first_act.shape[2]
        n_feature_blocks = n_blocks - 1  # last is target

        feature_block_map = {}
        for i in range(n_features):
            feature_block_map[i] = min(i * n_feature_blocks // max(n_features, 1), n_feature_blocks - 1)

        per_feature_vectors = []
        for feat_idx in range(n_features):
            block_idx = feature_block_map[feat_idx]
            feat_vector = []

            for layer_idx in sorted(activation_dict.keys()):
                act = activation_dict[layer_idx].cpu().float()
                # act: (batch, batch_items, n_blocks, emsize)
                # Average over batch and batch_items dims
                act_avg = act.mean(dim=(0, 1))  # (n_blocks, emsize)
                act_block = act_avg[min(block_idx, act_avg.shape[0] - 1)]  # (emsize,)

                feat_vector.append(float(act_block.mean()))
                feat_vector.append(float(act_block.std()))
                feat_vector.append(float(act_block.max()))
                # Sparsity
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
