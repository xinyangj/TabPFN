"""Compute per-block statistics on GPU before CPU transfer.

Replaces transferring ~1 GB of raw attention/activation tensors with
~20 KB of pre-computed statistics. The SignalProcessor then only needs
to do block→feature gather and cross-layer aggregation on these tiny arrays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class GPUStatsComputer:
    """Compute per-block summary statistics on GPU for fast CPU transfer.

    For enriched=False mode, computes:
    - between_items_attention: (n_layers, n_blocks, n_heads, 3) — entropy, max, var
    - between_features_attention: (n_layers, n_feat_blocks, n_heads, 6) — self, to_target,
      from_target, mean_to_others, mean_from_others, entropy
      + to_target_per_layer: (n_layers, n_feat_blocks) for cross-layer stats
    - mlp_activations: (n_layers, n_feat_blocks, 5) — mean, std, max, sparsity, l2_norm
    - attention_gradients: (n_layers, n_feat_blocks, n_heads, 2) — grad_to_target, mean_abs_grad
    """

    def __init__(self, enriched: bool = False):
        self.enriched = enriched

    @torch.no_grad()
    def compute(self, signals: dict[str, Any]) -> dict[str, Any]:
        """Compute all stats on GPU and return small CPU numpy arrays.

        Args:
            signals: Raw signals dict from SignalExtractor (tensors on GPU).

        Returns:
            Dict with pre-computed stats as numpy arrays, ready for
            SignalProcessor.process_from_stats().
        """
        result: dict[str, Any] = {}

        # Between-items attention stats
        items_attn = signals.get("between_items_attention")
        if items_attn and isinstance(items_attn, dict):
            stats = self._compute_items_attention_stats(items_attn)
            if stats is not None:
                result["items_attention_block_stats"] = stats

        # Between-features attention stats
        feat_attn = signals.get("between_features_attention")
        if feat_attn and isinstance(feat_attn, dict):
            stats = self._compute_features_attention_stats(feat_attn)
            if stats is not None:
                result.update(stats)

            # Attention rollout (enriched mode only, needs both attention types)
            if self.enriched:
                items_attn = signals.get("between_items_attention")
                if items_attn and isinstance(items_attn, dict):
                    rollout = self._compute_rollout(feat_attn, items_attn)
                    if rollout is not None:
                        result.update(rollout)

        # MLP activation stats
        mlp_act = signals.get("mlp_activations")
        if mlp_act and isinstance(mlp_act, dict):
            stats = self._compute_mlp_stats(mlp_act)
            if stats is not None:
                result["mlp_block_stats"] = stats

        # Small tensors: transfer directly to CPU
        for key in ("train_embeddings", "test_embeddings", "input_gradients"):
            val = signals.get(key)
            if val is not None and isinstance(val, torch.Tensor):
                result[key] = val.detach().cpu().float()

        # Attention gradients (small: between-features shape)
        attn_grads = signals.get("attention_gradients")
        if attn_grads and isinstance(attn_grads, dict):
            stats = self._compute_attn_gradient_stats(attn_grads)
            if stats is not None:
                result["attn_gradient_block_stats"] = stats
            else:
                # All grads None — store shape info so gather can emit zeros
                n_attn_layers = len(attn_grads)
                # Infer n_feat_blocks and n_heads from features attention stats
                fa_stats = result.get("features_attention_block_stats")
                if fa_stats is not None:
                    n_fb = fa_stats.shape[1]
                    n_h = fa_stats.shape[2]
                    result["attn_gradient_block_stats"] = np.zeros(
                        (n_attn_layers, n_fb, n_h, 2), dtype=np.float32
                    )

        return result

    def _compute_items_attention_stats(
        self, attention_dict: dict[str, torch.Tensor | None]
    ) -> np.ndarray | None:
        """Compute per-block stats for between-items attention on GPU.

        Returns: (n_layers, n_blocks, n_heads, n_stats) numpy array.
        """
        layer_tensors = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None and isinstance(attn, torch.Tensor):
                a = attn.float()
                a = torch.nan_to_num(a, nan=0.0)
                layer_tensors.append(a)

        if not layer_tensors:
            return None

        n_layers = len(layer_tensors)
        n_blocks = layer_tensors[0].shape[0]
        n_heads = layer_tensors[0].shape[3]
        n_stats = 3  # enriched=False: entropy, max, var

        device = layer_tensors[0].device
        block_stats = torch.zeros(n_layers, n_blocks, n_heads, n_stats, device=device)

        for li, at in enumerate(layer_tensors):
            # at: (n_blocks, n_items, n_items, n_heads)
            # Entropy: per-row entropy, mean over items
            row_ent = torch.special.entr(at).sum(dim=2)  # (n_blocks, n_items, n_heads)
            block_stats[li, :, :, 0] = row_ent.mean(dim=1)

            # Max and variance over flattened item dims
            af = at.reshape(n_blocks, -1, n_heads)  # (n_blocks, n_items*n_items, n_heads)
            block_stats[li, :, :, 1] = af.max(dim=1).values
            block_stats[li, :, :, 2] = af.var(dim=1)

        return block_stats.cpu().numpy()

    def _compute_features_attention_stats(
        self, attention_dict: dict[str, torch.Tensor | None]
    ) -> dict[str, np.ndarray] | None:
        """Compute per-block stats for between-features attention on GPU.

        Returns dict with:
          'features_attention_block_stats': (n_layers, n_feat_blocks, n_heads, 6)
          'features_to_target_per_layer': (n_layers, n_feat_blocks) for cross-layer
        """
        layer_tensors = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None and isinstance(attn, torch.Tensor):
                a = attn.float()
                a = torch.nan_to_num(a, nan=0.0)
                layer_tensors.append(a)

        if not layer_tensors:
            return None

        n_layers = len(layer_tensors)
        # Shape: (batch_items, n_blocks, n_blocks, n_heads)
        n_blocks = layer_tensors[0].shape[1]
        n_heads = layer_tensors[0].shape[3]
        target_block = n_blocks - 1
        n_feat_blocks = n_blocks - 1
        n_stats = 6  # enriched=False

        device = layer_tensors[0].device
        block_stats = torch.zeros(n_layers, n_feat_blocks, n_heads, n_stats, device=device)
        to_target_per_layer = torch.zeros(n_layers, n_feat_blocks, device=device)

        for li, at in enumerate(layer_tensors):
            # Mean over batch_items: (n_blocks, n_blocks, n_heads)
            mean_at = at.mean(dim=0)

            for h in range(n_heads):
                ah = mean_at[:, :, h]  # (n_blocks, n_blocks)
                feat_ah = ah[:n_feat_blocks, :]  # (n_feat_blocks, n_blocks)

                # 1. self-attention: diagonal
                block_stats[li, :, h, 0] = torch.diagonal(ah)[:n_feat_blocks]
                # 2. to_target
                block_stats[li, :, h, 1] = feat_ah[:, target_block]
                # 3. from_target
                block_stats[li, :, h, 2] = ah[target_block, :n_feat_blocks]

                if n_feat_blocks > 1:
                    out_to_feats = feat_ah[:, :n_feat_blocks]  # (n_fb, n_fb)
                    diag_vals = torch.diagonal(out_to_feats)
                    row_sums = out_to_feats.sum(dim=1)
                    col_sums = out_to_feats.sum(dim=0)
                    n_others = max(n_feat_blocks - 1, 1)

                    # 4. mean_to_others
                    block_stats[li, :, h, 3] = (row_sums - diag_vals) / n_others
                    # 5. mean_from_others
                    block_stats[li, :, h, 4] = (col_sums - diag_vals) / n_others

                # 6. entropy
                a_clamped = mean_at[:n_feat_blocks, :, h].clamp(min=1e-10)
                block_stats[li, :, h, 5] = -(a_clamped * a_clamped.log()).sum(dim=1)

            # to_target cross-layer: mean over heads
            to_target_per_layer[li] = mean_at[:n_feat_blocks, target_block, :].mean(dim=-1)

        return {
            "features_attention_block_stats": block_stats.cpu().numpy(),
            "features_to_target_per_layer": to_target_per_layer.cpu().numpy(),
        }

    def _compute_mlp_stats(
        self, activation_dict: dict[int, torch.Tensor]
    ) -> np.ndarray | None:
        """Compute per-block stats for MLP activations on GPU.

        Returns: (n_layers, n_feat_blocks, 5) numpy array.
        """
        sorted_layers = sorted(activation_dict.keys())
        if not sorted_layers:
            return None

        first_act = activation_dict[sorted_layers[0]]
        n_blocks = first_act.shape[2]
        n_feat_blocks = n_blocks - 1
        n_layers = len(sorted_layers)
        n_stats = 5  # enriched=False

        device = first_act.device
        block_stats = torch.zeros(n_layers, n_feat_blocks, n_stats, device=device)

        for li, layer_idx in enumerate(sorted_layers):
            act = activation_dict[layer_idx].float()
            # act: (batch, batch_items, n_blocks, emsize) → mean over batch+items
            act_avg = act.mean(dim=(0, 1))  # (n_blocks, emsize)
            feat_acts = act_avg[:n_feat_blocks]  # (n_feat_blocks, emsize)

            block_stats[li, :, 0] = feat_acts.mean(dim=1)        # mean
            block_stats[li, :, 1] = feat_acts.std(dim=1)          # std
            block_stats[li, :, 2] = feat_acts.max(dim=1).values   # max
            block_stats[li, :, 3] = (feat_acts.abs() < 0.01).float().mean(dim=1)  # sparsity
            block_stats[li, :, 4] = feat_acts.norm(dim=1)         # L2 norm

        return block_stats.cpu().numpy()

    def _compute_attn_gradient_stats(
        self, gradient_dict: dict[str, torch.Tensor | None]
    ) -> np.ndarray | None:
        """Compute per-block stats for attention gradients on GPU.

        Returns: (n_layers, n_feat_blocks, n_heads, 2) numpy array.
        If all gradients are None, returns zeros with shape inferred from
        other signals (to match the CPU code path).
        """
        sorted_keys = sorted(gradient_dict.keys())
        n_layers = len(sorted_keys)
        if n_layers == 0:
            return None

        layer_data = []
        for key in sorted_keys:
            grad = gradient_dict[key]
            if grad is not None and isinstance(grad, torch.Tensor):
                layer_data.append(grad.float())
            else:
                layer_data.append(None)

        # Find first non-None to get shape
        first_valid = next((t for t in layer_data if t is not None), None)
        if first_valid is None:
            # All gradients are None — return None and let caller handle it
            return None

        n_blocks = first_valid.shape[1]
        n_heads = first_valid.shape[3]
        n_feat_blocks = n_blocks - 1
        target_block = n_blocks - 1
        n_stats = 2  # enriched=False

        device = first_valid.device
        block_stats = torch.zeros(n_layers, n_feat_blocks, n_heads, n_stats, device=device)

        for li, grad_t in enumerate(layer_data):
            if grad_t is None:
                continue
            # grad_t: (batch_items, n_blocks, n_blocks, n_heads)
            g_avg = grad_t.mean(dim=0)  # (n_blocks, n_blocks, n_heads)

            for h in range(n_heads):
                gh = g_avg[:, :, h]  # (n_blocks, n_blocks)
                block_stats[li, :, h, 0] = gh[:n_feat_blocks, target_block]
                block_stats[li, :, h, 1] = gh[:n_feat_blocks, :].abs().mean(dim=1)

        return block_stats.cpu().numpy()

    def _compute_rollout(
        self,
        feat_attn_dict: dict[str, torch.Tensor | None],
        items_attn_dict: dict[str, torch.Tensor | None],
    ) -> dict[str, np.ndarray] | None:
        """Compute joint attention rollout across all layers on GPU.

        Combines both between-features and between-items attention into a
        joint N×N rollout matrix (N = n_items × n_blocks) following the
        GRN sequential rollout approach.  Per layer:

            A_feat = normalize(kron(feat_2d, J_blocks) + I)
            A_items = normalize(kron(J_items, item_2d) + I)
            A_layer = A_items @ A_feat
            rollout = A_layer @ rollout

        Extracts per-feature-block scores by averaging over items.

        Returns dict with:
          'rollout_final': (n_feat_blocks, n_blocks, 1) — per-block rollout
            to all block positions (averaged over items and heads)
          'rollout_mid': (n_feat_blocks, 1) — rollout_to_target at mid-layer
        """
        feat_layers = []
        items_layers = []
        for key in sorted(feat_attn_dict.keys()):
            fa = feat_attn_dict[key]
            ia = items_attn_dict.get(key)
            if fa is not None and isinstance(fa, torch.Tensor) and ia is not None and isinstance(ia, torch.Tensor):
                feat_layers.append(fa.float())
                items_layers.append(ia.float())

        if not feat_layers:
            return None

        n_layers = len(feat_layers)
        # Between-features: (batch_items=n_items, n_blocks, n_blocks, n_heads)
        n_blocks = feat_layers[0].shape[1]
        n_feat_blocks = n_blocks - 1
        target_block = n_blocks - 1
        # Between-items: (batch_blocks=n_blocks, n_items, n_items, n_heads)
        n_items = items_layers[0].shape[1]
        N = n_blocks * n_items
        mid_layer = n_layers // 2

        device = feat_layers[0].device
        I_N = torch.eye(N, device=device)

        rollout = I_N.clone()
        rollout_mid = None

        for li in range(n_layers):
            # Between-features: avg over batch_items and heads → (n_blocks, n_blocks)
            feat_2d = torch.nan_to_num(feat_layers[li], nan=0.0).mean(dim=(0, -1))
            # Between-items: avg over batch_blocks and heads → (n_items, n_items)
            item_2d = torch.nan_to_num(items_layers[li], nan=0.0).mean(dim=(0, -1))

            # Build N×N matrices via Kronecker products
            # A_feat: block-diagonal — each item sees the same feature attention
            # kron(I_items, feat_2d) would be block-diagonal
            # But GRN uses kron(feat_attn_averaged_over_items, J_blocks) — let's match GRN exactly
            # GRN: A_feat = kron(feat_2d, J_blocks) where feat_2d = (n_blocks, n_blocks)
            #       since between_features operates per item, Kronecker with J broadcasts
            # Actually the GRN treats: num_items = feat_attn.size(0) = n_blocks
            #                           num_feature_blocks = item_attn.size(0) = n_items
            # and N ordering = (block_b, item_i) → idx = block_b * n_items + item_i
            # So kron(feat_2d(n_blocks,n_blocks), ones(n_items,n_items)) works.

            J_items = torch.ones(n_items, n_items, device=device)
            J_blocks = torch.ones(n_blocks, n_blocks, device=device)

            A_feat = torch.kron(feat_2d, J_items)     # (N, N)
            A_items = torch.kron(J_blocks, item_2d)    # (N, N)

            # Add residual and normalize
            A_feat = A_feat + I_N
            A_feat = A_feat / (A_feat.sum(dim=-1, keepdim=True) + 1e-8)

            A_items = A_items + I_N
            A_items = A_items / (A_items.sum(dim=-1, keepdim=True) + 1e-8)

            # Sequential: items AFTER features
            A_layer = A_items @ A_feat
            rollout = A_layer @ rollout

            if li == mid_layer - 1:
                rollout_mid = self._extract_block_scores(
                    rollout, n_blocks, n_items, n_feat_blocks, target_block
                )

        if rollout_mid is None:
            rollout_mid = self._extract_block_scores(
                rollout, n_blocks, n_items, n_feat_blocks, target_block
            )

        # Extract final per-block scores
        # For each feature block b, average rollout[b*n_items+i, target_block*n_items+j]
        # over all items i and test positions j
        rollout_final = self._extract_block_rollout_matrix(
            rollout, n_blocks, n_items, n_feat_blocks
        )

        return {
            "rollout_final": rollout_final.cpu().numpy(),  # (n_feat_blocks, n_blocks, 1)
            "rollout_mid": rollout_mid.cpu().numpy(),      # (n_feat_blocks, 1)
        }

    @staticmethod
    def _extract_block_scores(
        rollout: torch.Tensor,
        n_blocks: int,
        n_items: int,
        n_feat_blocks: int,
        target_block: int,
    ) -> torch.Tensor:
        """Extract per-block to-target scores from N×N rollout matrix.

        Returns: (n_feat_blocks, 1) — mean rollout from each feature block
        to the target block, averaged over all item pairs.
        """
        scores = torch.zeros(n_feat_blocks, 1, device=rollout.device)
        target_start = target_block * n_items
        target_end = target_start + n_items

        for b in range(n_feat_blocks):
            src_start = b * n_items
            src_end = src_start + n_items
            # Average rollout from all items in block b to all items in target block
            scores[b, 0] = rollout[src_start:src_end, target_start:target_end].mean()

        return scores

    @staticmethod
    def _extract_block_rollout_matrix(
        rollout: torch.Tensor,
        n_blocks: int,
        n_items: int,
        n_feat_blocks: int,
    ) -> torch.Tensor:
        """Extract per-block-to-block rollout from N×N matrix.

        Returns: (n_feat_blocks, n_blocks, 1) — mean rollout from each
        source feature block to each destination block, averaged over items.
        """
        result = torch.zeros(n_feat_blocks, n_blocks, 1, device=rollout.device)
        for src_b in range(n_feat_blocks):
            src_start = src_b * n_items
            src_end = src_start + n_items
            for dst_b in range(n_blocks):
                dst_start = dst_b * n_items
                dst_end = dst_start + n_items
                result[src_b, dst_b, 0] = rollout[src_start:src_end, dst_start:dst_end].mean()

        return result
