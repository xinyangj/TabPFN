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
        n_train = signals.get("n_train", 0)

        # Between-items attention stats
        items_attn = signals.get("between_items_attention")
        if items_attn and isinstance(items_attn, dict):
            stats = self._compute_items_attention_stats(items_attn, n_train=n_train)
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
            mlp_result = self._compute_mlp_stats(mlp_act)
            if mlp_result is not None:
                result["mlp_block_stats"] = mlp_result["block_stats"]
                if "cross_layer" in mlp_result:
                    result["mlp_cross_layer"] = mlp_result["cross_layer"]

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

        # Hidden state gradients (enriched mode only)
        if self.enriched:
            hidden_grads = signals.get("hidden_state_gradients")
            if hidden_grads and isinstance(hidden_grads, dict):
                hg_result = self._compute_hidden_grad_stats(hidden_grads)
                if hg_result is not None:
                    result.update(hg_result)

        # Value contributions (enriched mode only)
        if self.enriched:
            val_contribs = signals.get("value_contributions")
            if val_contribs and isinstance(val_contribs, dict):
                vc_result = self._compute_value_contribution_stats(val_contribs)
                if vc_result is not None:
                    result.update(vc_result)

                # Logit attribution scalars (needs decoder Jacobian)
                dec_jac = signals.get("decoder_jacobian")
                if dec_jac is not None:
                    logit_attr = self._compute_logit_attribution(
                        val_contribs, dec_jac
                    )
                    if logit_attr is not None:
                        result["logit_attribution"] = logit_attr

        return result

    def _compute_items_attention_stats(
        self, attention_dict: dict[str, torch.Tensor | None],
        *, n_train: int = 0,
    ) -> np.ndarray | None:
        """Compute per-block stats for between-items attention on GPU.

        Fully vectorized across layers, blocks, and heads — no Python loops.

        Returns: (n_layers, n_blocks, n_heads, n_stats) numpy array.
        n_stats = 9 if enriched, 3 otherwise.
        """
        layer_tensors = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None and isinstance(attn, torch.Tensor):
                layer_tensors.append(torch.nan_to_num(attn.float(), nan=0.0))

        if not layer_tensors:
            return None

        # Stack all layers: (n_layers, n_blocks, n_items, n_items, n_heads)
        all_at = torch.stack(layer_tensors)
        n_layers, n_blocks, n_items, _, n_heads = all_at.shape
        n_stats = 9 if self.enriched else 3
        has_split = n_train > 0 and n_train < n_items

        device = all_at.device
        block_stats = torch.zeros(n_layers, n_blocks, n_heads, n_stats, device=device)

        # 1. Entropy: per-row entropy, mean over items
        row_ent = torch.special.entr(all_at).sum(dim=3)  # (L, nb, ni, H)
        block_stats[:, :, :, 0] = row_ent.mean(dim=2)

        # 2-3. Max and variance over flattened item dims
        af = all_at.reshape(n_layers, n_blocks, -1, n_heads)  # (L, nb, ni*ni, H)
        max_vals = af.max(dim=2).values  # (L, nb, H)
        block_stats[:, :, :, 1] = max_vals
        block_stats[:, :, :, 2] = af.var(dim=2)

        if self.enriched and has_split:
            # 4. train_to_test
            t2t = all_at[:, :, :n_train, n_train:, :].mean(dim=(2, 3))  # (L, nb, H)
            block_stats[:, :, :, 3] = t2t
            # 5. test_to_train
            block_stats[:, :, :, 4] = all_at[:, :, n_train:, :n_train, :].mean(dim=(2, 3))
            # 6. self_train
            self_train = all_at[:, :, :n_train, :n_train, :].mean(dim=(2, 3))
            block_stats[:, :, :, 5] = self_train
            # 7. self_test
            block_stats[:, :, :, 6] = all_at[:, :, n_train:, n_train:, :].mean(dim=(2, 3))
            # 8. train_test_ratio
            block_stats[:, :, :, 7] = t2t / (self_train + 1e-10)
            # 9. concentration: max / mean_row_sum
            row_sums = all_at.sum(dim=3).mean(dim=2)  # (L, nb, H)
            block_stats[:, :, :, 8] = max_vals / (row_sums + 1e-10)

        return block_stats.cpu().numpy()

    def _compute_features_attention_stats(
        self, attention_dict: dict[str, torch.Tensor | None]
    ) -> dict[str, np.ndarray] | None:
        """Compute per-block stats for between-features attention on GPU.

        Returns dict with:
          'features_attention_block_stats': (n_layers, n_feat_blocks, n_heads, n_stats)
              n_stats = 15 if enriched, 6 otherwise
          'features_to_target_per_layer': (n_layers, n_feat_blocks) for cross-layer
        """
        layer_tensors = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None and isinstance(attn, torch.Tensor):
                layer_tensors.append(torch.nan_to_num(attn.float(), nan=0.0))

        if not layer_tensors:
            return None

        # Stack all layers: (n_layers, batch_items, n_blocks, n_blocks, n_heads)
        all_at = torch.stack(layer_tensors)
        n_layers = all_at.shape[0]
        n_blocks = all_at.shape[2]
        n_heads = all_at.shape[4]
        target_block = n_blocks - 1
        n_feat_blocks = n_blocks - 1
        n_stats = 15 if self.enriched else 6

        device = all_at.device

        # Mean over batch_items: (n_layers, n_blocks, n_blocks, n_heads)
        mean_at = all_at.mean(dim=1)
        del all_at

        block_stats = torch.zeros(n_layers, n_feat_blocks, n_heads, n_stats, device=device)

        diag_idx = torch.arange(n_feat_blocks, device=device)
        feat_feat = mean_at[:, :n_feat_blocks, :n_feat_blocks, :]

        # 1. self-attention (diagonal)
        block_stats[:, :, :, 0] = feat_feat[:, diag_idx, diag_idx, :]

        # 2. to_target
        to_target = mean_at[:, :n_feat_blocks, target_block, :]
        block_stats[:, :, :, 1] = to_target

        # 3. from_target
        from_target = mean_at[:, target_block, :n_feat_blocks, :]
        block_stats[:, :, :, 2] = from_target

        if n_feat_blocks > 1:
            diag_vals = feat_feat[:, diag_idx, diag_idx, :]
            row_sums = feat_feat.sum(dim=2)
            col_sums = feat_feat.sum(dim=1)
            n_others = max(n_feat_blocks - 1, 1)

            mean_to_others = (row_sums - diag_vals) / n_others
            block_stats[:, :, :, 3] = mean_to_others
            block_stats[:, :, :, 4] = (col_sums - diag_vals) / n_others

            if self.enriched:
                masked_feat = feat_feat.clone()
                masked_feat[:, diag_idx, diag_idx, :] = float('-inf')
                zero_diag_feat = feat_feat.clone()
                zero_diag_feat[:, diag_idx, diag_idx, :] = 0.0

                block_stats[:, :, :, 6] = masked_feat.max(dim=2).values

                out_sum = zero_diag_feat.sum(dim=2)
                out_sq_sum = (zero_diag_feat ** 2).sum(dim=2)
                out_mean = out_sum / n_others
                block_stats[:, :, :, 7] = ((out_sq_sum / n_others - out_mean ** 2).clamp(min=0)).sqrt()

                block_stats[:, :, :, 8] = masked_feat.max(dim=1).values

                in_sum = zero_diag_feat.sum(dim=1)
                in_sq_sum = (zero_diag_feat ** 2).sum(dim=1)
                in_mean = in_sum / n_others
                block_stats[:, :, :, 9] = ((in_sq_sum / n_others - in_mean ** 2).clamp(min=0)).sqrt()

                asym = feat_feat - feat_feat.transpose(1, 2)
                asym[:, diag_idx, diag_idx, :] = 0.0
                block_stats[:, :, :, 10] = asym.sum(dim=2) / n_others

                asym_abs = asym.abs()
                asym_abs[:, diag_idx, diag_idx, :] = 0.0
                block_stats[:, :, :, 11] = asym_abs.max(dim=2).values

                del masked_feat, zero_diag_feat, asym, asym_abs

        feat_to_all = mean_at[:, :n_feat_blocks, :, :]
        a_clamped = feat_to_all.clamp(min=1e-10)
        block_stats[:, :, :, 5] = -(a_clamped * a_clamped.log()).sum(dim=2)

        if self.enriched:
            block_stats[:, :, :, 12] = (feat_to_all <= to_target.unsqueeze(2)).float().mean(dim=2)

            all_to_feat = mean_at[:, :, :n_feat_blocks, :]
            in_col = all_to_feat.transpose(1, 2)
            block_stats[:, :, :, 13] = (in_col <= from_target.unsqueeze(2)).float().mean(dim=2)

            block_stats[:, :, :, 14] = to_target - block_stats[:, :, :, 3]

        to_target_per_layer = mean_at[:, :n_feat_blocks, target_block, :].mean(dim=-1)

        return {
            "features_attention_block_stats": block_stats.cpu().numpy(),
            "features_to_target_per_layer": to_target_per_layer.cpu().numpy(),
        }

    def _compute_mlp_stats(
        self, activation_dict: dict[int, torch.Tensor]
    ) -> dict[str, np.ndarray] | None:
        """Compute per-block stats for MLP activations on GPU.

        Fully vectorized across layers — no Python loops.

        Returns dict with:
          'block_stats': (n_layers, n_feat_blocks, n_stats) — n_stats=10 if enriched, 5 otherwise
          'cross_layer': (n_feat_blocks, 3) — only if enriched (norm_trend, cosine_first_last, norm_ratio)
        """
        sorted_layers = sorted(activation_dict.keys())
        if not sorted_layers:
            return None

        first_act = activation_dict[sorted_layers[0]]
        n_blocks = first_act.shape[2]
        n_feat_blocks = n_blocks - 1
        target_block = n_blocks - 1
        n_layers = len(sorted_layers)
        n_stats = 10 if self.enriched else 5
        device = first_act.device

        # Stack all layers: (L, batch, batch_items, n_blocks, emsize)
        all_act = torch.stack([activation_dict[k].float() for k in sorted_layers])
        act_avg = all_act.mean(dim=(1, 2))  # (L, n_blocks, emsize)
        del all_act
        feat_acts = act_avg[:, :n_feat_blocks, :]  # (L, nfb, emsize)

        block_stats = torch.zeros(n_layers, n_feat_blocks, n_stats, device=device)

        block_stats[:, :, 0] = feat_acts.mean(dim=2)
        block_stats[:, :, 1] = feat_acts.std(dim=2)
        block_stats[:, :, 2] = feat_acts.max(dim=2).values
        block_stats[:, :, 3] = (feat_acts.abs() < 0.01).float().mean(dim=2)
        norms = feat_acts.norm(dim=2)
        block_stats[:, :, 4] = norms

        if self.enriched:
            target_acts = act_avg[:, target_block, :]  # (L, emsize)
            target_norms = target_acts.norm(dim=1) + 1e-10
            dot_products = (feat_acts * target_acts.unsqueeze(1)).sum(dim=2)
            block_stats[:, :, 5] = dot_products / (norms * target_norms.unsqueeze(1) + 1e-10)

            all_blocks_mean = feat_acts.mean(dim=1, keepdim=True)
            block_stats[:, :, 6] = (feat_acts - all_blocks_mean).norm(dim=2)
            block_stats[:, :, 7] = feat_acts.min(dim=2).values

            stds = block_stats[:, :, 1]
            means = block_stats[:, :, 0]
            centered = feat_acts - means.unsqueeze(2)
            safe_stds = torch.where(stds > 1e-10, stds, torch.ones_like(stds))
            skew = (centered ** 3).mean(dim=2) / (safe_stds ** 3)
            block_stats[:, :, 8] = torch.where(stds > 1e-10, skew, torch.zeros_like(skew))
            block_stats[:, :, 9] = (feat_acts > 0).float().mean(dim=2)

        result = {"block_stats": block_stats.cpu().numpy()}

        if self.enriched and n_layers > 0:
            norms_all = block_stats[:, :, 4]
            cross = torch.zeros(n_feat_blocks, 3, device=device)

            if n_layers > 1:
                x_arr = torch.arange(n_layers, dtype=torch.float32, device=device)
                x_mean = x_arr.mean()
                x_var = ((x_arr - x_mean) ** 2).sum()
                if x_var > 0:
                    cross[:, 0] = (
                        (x_arr.unsqueeze(1) - x_mean) *
                        (norms_all - norms_all.mean(dim=0, keepdim=True))
                    ).sum(dim=0) / x_var

            first_acts = feat_acts[0]
            last_acts = feat_acts[-1]
            dot = (first_acts * last_acts).sum(dim=1)
            fn = first_acts.norm(dim=1) + 1e-10
            ln = last_acts.norm(dim=1) + 1e-10
            cross[:, 1] = dot / (fn * ln)
            cross[:, 2] = ln / (fn + 1e-10)

            result["cross_layer"] = cross.cpu().numpy()

        return result

    def _compute_attn_gradient_stats(
        self, gradient_dict: dict[str, torch.Tensor | None]
    ) -> np.ndarray | None:
        """Compute per-block stats for attention gradients on GPU.

        Returns: (n_layers, n_feat_blocks, n_heads, 2) numpy array.
        If all gradients are None, returns None.
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

        first_valid = next((t for t in layer_data if t is not None), None)
        if first_valid is None:
            return None

        n_blocks = first_valid.shape[1]
        n_heads = first_valid.shape[3]
        n_feat_blocks = n_blocks - 1
        target_block = n_blocks - 1

        device = first_valid.device
        block_stats = torch.zeros(n_layers, n_feat_blocks, n_heads, 2, device=device)

        for li, grad_t in enumerate(layer_data):
            if grad_t is None:
                continue
            g_avg = grad_t.mean(dim=0)  # (n_blocks, n_blocks, n_heads)
            # Vectorized across heads
            block_stats[li, :, :, 0] = g_avg[:n_feat_blocks, target_block, :]
            block_stats[li, :, :, 1] = g_avg[:n_feat_blocks, :, :].abs().mean(dim=1)

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

    @torch.no_grad()
    def _compute_hidden_grad_stats(
        self, hidden_grads: dict[int, torch.Tensor]
    ) -> dict[str, np.ndarray] | None:
        """Compute per-block stats and raw vectors from hidden state gradients.

        Hidden state gradients have shape (batch, n_items, n_blocks, emsize)
        per layer. We compute:
        1. Per-layer stats (6 per block): norm, mean, abs_max, sparsity, std, pos_frac
        2. Raw 192d vectors for 3 key layers (first, mid, last)
        3. Layer-averaged raw 192d vector

        Returns dict with:
          'hidden_grad_block_stats': (n_layers, n_feat_blocks, 6) numpy
          'hidden_grad_raw_key': (3, n_feat_blocks, emsize) numpy  [layers 0, mid, last]
          'hidden_grad_raw_avg': (n_feat_blocks, emsize) numpy
        """
        sorted_keys = sorted(hidden_grads.keys())
        if not sorted_keys:
            return None

        n_layers = len(sorted_keys)

        # Stack all layers: each (batch, n_items, n_blocks, emsize)
        all_grads = torch.stack([hidden_grads[k].float() for k in sorted_keys])
        # → (L, batch, n_items, n_blocks, emsize)

        # Mean over batch and items → (L, n_blocks, emsize)
        grad_avg = all_grads.mean(dim=(1, 2))
        del all_grads

        n_blocks = grad_avg.shape[1]
        n_feat_blocks = n_blocks - 1
        emsize = grad_avg.shape[2]

        # Feature blocks only (exclude target block)
        feat_grads = grad_avg[:, :n_feat_blocks, :]  # (L, nfb, emsize)

        device = feat_grads.device
        n_stats = 6
        block_stats = torch.zeros(n_layers, n_feat_blocks, n_stats, device=device)

        # All stats vectorized across layers and blocks
        block_stats[:, :, 0] = feat_grads.norm(dim=2)                           # 1. grad_norm
        block_stats[:, :, 1] = feat_grads.mean(dim=2)                           # 2. grad_mean
        block_stats[:, :, 2] = feat_grads.abs().max(dim=2).values               # 3. grad_abs_max
        block_stats[:, :, 3] = (feat_grads.abs() < 0.01).float().mean(dim=2)    # 4. grad_sparsity
        block_stats[:, :, 4] = feat_grads.std(dim=2)                            # 5. grad_std
        block_stats[:, :, 5] = (feat_grads > 0).float().mean(dim=2)             # 6. grad_pos_frac

        # Raw gradients for 3 key layers: first (0), mid (L//2), last (L-1)
        mid = n_layers // 2
        last = n_layers - 1
        key_indices = [0, mid, last]
        raw_key = feat_grads[key_indices]  # (3, nfb, emsize)

        # Layer-averaged raw gradient
        raw_avg = feat_grads.mean(dim=0)  # (nfb, emsize)

        return {
            "hidden_grad_block_stats": block_stats.cpu().numpy(),
            "hidden_grad_raw_key": raw_key.cpu().numpy(),
            "hidden_grad_raw_avg": raw_avg.cpu().numpy(),
        }

    def _compute_value_contribution_stats(
        self,
        contributions: dict[int, torch.Tensor],
    ) -> dict[str, np.ndarray] | None:
        """Compute per-block stats for attention value contributions on GPU.

        Each contribution tensor has shape ``(n_items, n_feat_blocks, emsize)``.
        Returns stats, raw key layers, and raw layer average — same structure
        as hidden state gradient stats.
        """
        sorted_keys = sorted(contributions.keys())
        n_layers = len(sorted_keys)
        if n_layers == 0:
            return None

        # Stack: (L, n_items, n_feat_blocks, emsize)
        stacked = torch.stack([contributions[k].float() for k in sorted_keys])
        # Mean over items → (L, n_feat_blocks, emsize)
        feat_contribs = stacked.mean(dim=1)
        n_feat_blocks = feat_contribs.shape[1]
        device = feat_contribs.device

        # Per-layer stats: 6 per layer per block
        n_stats = 6
        block_stats = torch.zeros(n_layers, n_feat_blocks, n_stats, device=device)
        block_stats[:, :, 0] = feat_contribs.norm(dim=2)
        block_stats[:, :, 1] = feat_contribs.mean(dim=2)
        block_stats[:, :, 2] = feat_contribs.abs().max(dim=2).values
        block_stats[:, :, 3] = feat_contribs.std(dim=2)
        block_stats[:, :, 4] = (feat_contribs > 0).float().mean(dim=2)
        block_stats[:, :, 5] = (feat_contribs.abs() < 0.01).float().mean(dim=2)

        # Raw contributions at 3 key layers
        mid = n_layers // 2
        last = n_layers - 1
        raw_key = feat_contribs[[0, mid, last]]  # (3, nfb, emsize)
        raw_avg = feat_contribs.mean(dim=0)  # (nfb, emsize)

        return {
            "value_contrib_block_stats": block_stats.cpu().numpy(),
            "value_contrib_raw_key": raw_key.cpu().numpy(),
            "value_contrib_raw_avg": raw_avg.cpu().numpy(),
        }

    def _compute_logit_attribution(
        self,
        contributions: dict[int, torch.Tensor],
        decoder_jacobian: torch.Tensor,
    ) -> np.ndarray | None:
        """Compute per-layer logit attribution scalars.

        Projects each layer's per-block value contribution through the decoder
        Jacobian to get a signed scalar measuring how much each block's
        contribution at each layer affects the prediction.

        Returns shape ``(n_layers, n_feat_blocks)``.
        """
        sorted_keys = sorted(contributions.keys())
        n_layers = len(sorted_keys)
        if n_layers == 0:
            return None

        # Stack: (L, n_items, n_feat_blocks, emsize)
        stacked = torch.stack([contributions[k].float() for k in sorted_keys])
        # decoder_jacobian: (n_items, emsize)
        J = decoder_jacobian.float()

        # Dot product: J[i] · contrib[l, i, b] → scalar per layer, item, block
        # Then mean over items → (L, n_feat_blocks)
        # J shape: (n_items, emsize) → (1, n_items, 1, emsize)
        attr = torch.einsum("ie, life -> lf", J, stacked)  # sum over items and emsize
        # Normalize by n_items
        n_items = stacked.shape[1]
        attr = attr / n_items

        return attr.cpu().numpy()
