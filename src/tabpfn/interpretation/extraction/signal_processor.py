"""Process raw TabPFN signals into fixed-size per-feature vectors.

Transforms variable-size attention matrices, embeddings, gradients,
and activations into a fixed-dimensional feature vector for each input
feature, suitable for the interpretation model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _extract_block_scores_cpu(
    rollout: np.ndarray,
    n_blocks: int,
    n_items: int,
    n_feat_blocks: int,
    target_block: int,
) -> np.ndarray:
    """Extract per-block to-target scores from N×N rollout (CPU numpy).

    Returns (n_feat_blocks, 1).
    """
    scores = np.zeros((n_feat_blocks, 1), dtype=np.float32)
    target_start = target_block * n_items
    target_end = target_start + n_items

    for b in range(n_feat_blocks):
        src_start = b * n_items
        src_end = src_start + n_items
        scores[b, 0] = rollout[src_start:src_end, target_start:target_end].mean()

    return scores


class SignalProcessor:
    """Process extracted TabPFN signals into per-feature feature vectors.

    Takes the raw signals from SignalExtractor and computes per-feature
    statistics that form the input to the interpretation model.

    The output is a matrix of shape (n_features, D_total) where D_total
    is the total dimension of all processed signal categories.

    Parameters
    ----------
    enriched : bool, default True
        If True, compute all enriched statistics (15 stats/head for attention,
        etc.). If False, compute only legacy statistics matching the v2 cache
        format (6 stats/head for attention, etc.) for backward compatibility.

    Examples
    --------
    >>> processor = SignalProcessor()
    >>> feature_vectors = processor.process(signals)
    >>> print(feature_vectors.shape)  # (n_features, D_total)
    """

    def __init__(self, *, enriched: bool = True) -> None:
        self.enriched = enriched

    def process(
        self,
        signals: dict[str, Any],
        *,
        signal_categories: set[str] | None = None,
    ) -> np.ndarray:
        """Process all signals into per-feature feature vectors.

        Parameters
        ----------
        signals : dict
            Output from SignalExtractor.extract().
        signal_categories : set of str, optional
            Which signal categories to include. If *None*, all are included.
            Valid categories: ``"between_features_attention"``,
            ``"between_items_attention"``, ``"embeddings"``, ``"gradients"``,
            ``"mlp_activations"``.

        Returns
        -------
        np.ndarray
            Per-feature feature matrix of shape (n_features, D_total).
        """
        all_cats = {
            "between_features_attention",
            "between_items_attention",
            "embeddings",
            "gradients",
            "mlp_activations",
            "items_attention_gradients",
        }
        cats = signal_categories if signal_categories is not None else all_cats

        n_features = signals["n_features"]

        feature_parts = []

        # 1. Between-features attention features
        if "between_features_attention" in cats:
            items_attn_for_rollout = signals.get("between_items_attention") if self.enriched else None
            feat_attn = self._process_between_features_attention(
                signals["between_features_attention"], n_features,
                items_attention_dict=items_attn_for_rollout,
            )
            if feat_attn is not None:
                feature_parts.append(feat_attn)

        # 2. Between-items attention features
        n_train = signals.get("n_train", 0)
        if "between_items_attention" in cats:
            item_attn = self._process_between_items_attention(
                signals["between_items_attention"], n_features, n_train=n_train
            )
            if item_attn is not None:
                feature_parts.append(item_attn)

        # 3. Embedding features
        if "embeddings" in cats:
            emb_feats = self._process_embeddings(signals, n_features)
            if emb_feats is not None:
                feature_parts.append(emb_feats)

        # 4. Gradient features
        if "gradients" in cats:
            grad_feats = self._process_gradients(signals, n_features)
            if grad_feats is not None:
                feature_parts.append(grad_feats)

        # 5. MLP activation features
        if "mlp_activations" in cats:
            act_feats = self._process_mlp_activations(
                signals["mlp_activations"], n_features
            )
            if act_feats is not None:
                feature_parts.append(act_feats)

        # 6. Items attention gradient features
        if "items_attention_gradients" in cats:
            items_grad_feats = self._process_items_attention_gradients(
                signals, n_features, n_train=n_train
            )
            if items_grad_feats is not None:
                feature_parts.append(items_grad_feats)

        if not feature_parts:
            raise ValueError("No signals could be processed.")

        result = np.concatenate(feature_parts, axis=1)
        # Clean up any NaN/Inf values
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def process_from_stats(
        self,
        gpu_stats: dict[str, Any],
        n_features: int,
        *,
        signal_categories: set[str] | None = None,
    ) -> np.ndarray:
        """Process pre-computed GPU stats into per-feature feature vectors.

        This is the fast path: accepts pre-computed per-block statistics from
        GPUStatsComputer.compute() and does only block→feature gather and
        cross-layer aggregation on tiny numpy arrays.

        Parameters
        ----------
        gpu_stats : dict
            Output from GPUStatsComputer.compute(). Contains pre-computed
            block-level stats as small numpy arrays.
        n_features : int
            Number of input features.
        signal_categories : set of str, optional
            Which signal categories to include. If *None*, all available are used.

        Returns
        -------
        np.ndarray
            Per-feature feature matrix of shape (n_features, D_total).
        """
        all_cats = {
            "between_features_attention",
            "between_items_attention",
            "embeddings",
            "gradients",
            "mlp_activations",
        }
        cats = signal_categories if signal_categories is not None else all_cats

        feature_parts = []

        # 1. Between-features attention (from pre-computed block stats)
        if "between_features_attention" in cats and "features_attention_block_stats" in gpu_stats:
            fv = self._gather_features_attention_stats(
                gpu_stats["features_attention_block_stats"],
                gpu_stats.get("features_to_target_per_layer"),
                n_features,
            )
            if fv is not None:
                feature_parts.append(fv)

            # Attention rollout (enriched mode only)
            if self.enriched and "rollout_final" in gpu_stats and "rollout_mid" in gpu_stats:
                block_stats = gpu_stats["features_attention_block_stats"]
                n_feat_blocks = block_stats.shape[1]
                bi_arr = np.array([min(i * n_feat_blocks // n_features, n_feat_blocks - 1)
                                   for i in range(n_features)])
                rollout_fv = self._gather_rollout_stats(
                    gpu_stats["rollout_final"],
                    gpu_stats["rollout_mid"],
                    n_features,
                    bi_arr,
                )
                feature_parts.append(rollout_fv)

        # 2. Between-items attention (from pre-computed block stats)
        if "between_items_attention" in cats and "items_attention_block_stats" in gpu_stats:
            fv = self._gather_items_attention_stats(
                gpu_stats["items_attention_block_stats"],
                n_features,
            )
            if fv is not None:
                feature_parts.append(fv)

        # 3. Embedding features
        if "embeddings" in cats:
            emb_feats = self._process_embeddings(gpu_stats, n_features)
            if emb_feats is not None:
                feature_parts.append(emb_feats)

        # 4. Gradient features (input gradients + attention gradient stats)
        if "gradients" in cats:
            fv = self._gather_gradient_stats(gpu_stats, n_features)
            if fv is not None:
                feature_parts.append(fv)

        # 5. MLP activation features (from pre-computed block stats)
        if "mlp_activations" in cats and "mlp_block_stats" in gpu_stats:
            fv = self._gather_mlp_stats(
                gpu_stats["mlp_block_stats"],
                n_features,
            )
            if fv is not None:
                feature_parts.append(fv)

        if not feature_parts:
            raise ValueError("No stats could be processed.")

        result = np.concatenate(feature_parts, axis=1)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    # ── Gather methods for process_from_stats ─────────────────────────

    def _gather_features_attention_stats(
        self,
        block_stats: np.ndarray,
        to_target_per_layer: np.ndarray | None,
        n_features: int,
    ) -> np.ndarray | None:
        """Gather pre-computed features attention stats to per-feature vectors.

        block_stats: (n_layers, n_feat_blocks, n_heads, n_stats)
        to_target_per_layer: (n_layers, n_feat_blocks)
        """
        n_layers, n_feat_blocks, n_heads, n_stats = block_stats.shape

        bi_arr = np.array([min(i * n_feat_blocks // n_features, n_feat_blocks - 1)
                           for i in range(n_features)])

        # Gather per-feature
        gathered = block_stats[:, bi_arr, :, :]  # (n_layers, n_features, n_heads, n_stats)
        per_head_stats = gathered.transpose(1, 0, 2, 3).reshape(n_features, -1)

        # Cross-layer stats
        cross_layer = np.zeros((n_features, 3), dtype=np.float32)
        if to_target_per_layer is not None:
            to_target_feat = to_target_per_layer[:, bi_arr]  # (n_layers, n_features)
            cross_layer[:, 0] = to_target_feat.mean(axis=0)
            cross_layer[:, 1] = to_target_feat.std(axis=0)
            if n_layers > 1:
                x_arr = np.arange(n_layers, dtype=np.float32)
                x_mean = x_arr.mean()
                x_var = ((x_arr - x_mean) ** 2).sum()
                if x_var > 0:
                    cross_layer[:, 2] = (
                        (x_arr[:, None] - x_mean) *
                        (to_target_feat - to_target_feat.mean(axis=0))
                    ).sum(axis=0) / x_var

        return np.concatenate([per_head_stats, cross_layer], axis=1).astype(np.float32)

    def _gather_items_attention_stats(
        self,
        block_stats: np.ndarray,
        n_features: int,
    ) -> np.ndarray | None:
        """Gather pre-computed items attention stats to per-feature vectors.

        block_stats: (n_layers, n_blocks, n_heads, n_stats)
        """
        n_blocks_attn = block_stats.shape[1]

        bi_arr = np.array([min(i * n_blocks_attn // max(n_features, 1), n_blocks_attn - 1)
                           for i in range(n_features)])

        gathered = block_stats[:, bi_arr, :, :]
        return gathered.transpose(1, 0, 2, 3).reshape(n_features, -1).astype(np.float32)

    def _gather_mlp_stats(
        self,
        block_stats: np.ndarray,
        n_features: int,
    ) -> np.ndarray | None:
        """Gather pre-computed MLP stats to per-feature vectors.

        block_stats: (n_layers, n_feat_blocks, n_stats)
        """
        n_feat_blocks = block_stats.shape[1]

        bi_arr = np.array([min(i * n_feat_blocks // max(n_features, 1), n_feat_blocks - 1)
                           for i in range(n_features)])

        gathered = block_stats[:, bi_arr, :]
        return gathered.transpose(1, 0, 2).reshape(n_features, -1).astype(np.float32)

    def _gather_gradient_stats(
        self,
        gpu_stats: dict[str, Any],
        n_features: int,
    ) -> np.ndarray | None:
        """Process gradient stats from GPU pre-computation.

        Input gradients are small CPU tensors; attention gradient stats are
        pre-computed per-block arrays.
        """
        parts = []

        # Input gradients (already on CPU as torch tensor)
        input_grads = gpu_stats.get("input_gradients")
        if input_grads is not None:
            grads_np = input_grads.numpy() if isinstance(input_grads, torch.Tensor) else np.asarray(input_grads, dtype=np.float32)
            if grads_np.ndim == 3:
                grads_np = grads_np[:, 0, :]  # (seq_len, n_features)

            n_grad_feats = min(grads_np.shape[1], n_features)
            abs_grads = np.abs(grads_np[:, :n_grad_feats])

            input_stats = np.zeros((n_features, 4), dtype=np.float32)
            input_stats[:n_grad_feats, 0] = abs_grads.mean(axis=0)
            input_stats[:n_grad_feats, 1] = abs_grads.max(axis=0)
            input_stats[:n_grad_feats, 2] = grads_np[:, :n_grad_feats].std(axis=0)
            pos_frac = (grads_np[:, :n_grad_feats] > 0).mean(axis=0)
            input_stats[:n_grad_feats, 3] = np.maximum(pos_frac, 1.0 - pos_frac)
            if n_grad_feats < n_features:
                input_stats[n_grad_feats:, 3] = 0.5
            parts.append(input_stats)

        # Attention gradient block stats (pre-computed)
        attn_grad_stats = gpu_stats.get("attn_gradient_block_stats")
        if attn_grad_stats is not None:
            n_feat_blocks = attn_grad_stats.shape[1]
            bi_arr = np.array([min(i * n_feat_blocks // max(n_features, 1), n_feat_blocks - 1)
                               for i in range(n_features)])
            gathered = attn_grad_stats[:, bi_arr, :, :]
            attn_grad_flat = gathered.transpose(1, 0, 2, 3).reshape(n_features, -1)
            parts.append(attn_grad_flat)

        if not parts:
            return None
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _gather_rollout_stats(
        self,
        rollout_final: np.ndarray,
        rollout_mid: np.ndarray,
        n_features: int,
        bi_arr: np.ndarray,
    ) -> np.ndarray:
        """Gather joint attention rollout stats into per-feature vectors.

        Parameters
        ----------
        rollout_final : (n_feat_blocks, n_blocks, 1)
            Block-to-block rollout (joint features+items), averaged over items.
        rollout_mid : (n_feat_blocks, 1)
            Rollout to target at mid-layer.
        n_features : int
        bi_arr : (n_features,)
            Feature-to-block mapping.

        Returns
        -------
        np.ndarray of shape (n_features, 8)
            Per-feature rollout features.
        """
        n_blocks = rollout_final.shape[1]
        target_block = n_blocks - 1
        n_fb = rollout_final.shape[0]

        result = np.zeros((n_features, 8), dtype=np.float32)

        rf = rollout_final[:, :, 0]  # (n_feat_blocks, n_blocks)

        # R1: rollout_to_target
        to_target = rf[bi_arr, target_block]  # (n_features,)
        result[:, 0] = to_target

        # R2: rollout_from_target (symmetric proxy — true from_target
        # would need the full N×N matrix including target rows)
        result[:, 1] = to_target

        # R3: rollout_self (diagonal)
        result[:, 2] = rf[bi_arr, bi_arr]

        # R4: rollout_rank — percentile of this block's to_target
        all_block_to_target = rf[:, target_block]  # (n_feat_blocks,)
        block_val = all_block_to_target[bi_arr]  # (n_features,)
        result[:, 3] = np.array([
            (all_block_to_target <= block_val[i]).mean()
            for i in range(n_features)
        ])

        # R5: rollout_contrast — this block minus mean of others
        total = all_block_to_target.sum()
        n_others = max(n_fb - 1, 1)
        others_mean = (total - all_block_to_target[bi_arr]) / n_others
        result[:, 4] = to_target - others_mean

        # R6: rollout_entropy — entropy of rollout distribution
        rf_for_feat = rf[bi_arr]  # (n_features, n_blocks)
        rf_clamped = np.clip(rf_for_feat, 1e-10, None)
        result[:, 5] = -(rf_clamped * np.log(rf_clamped)).sum(axis=1)

        # R7: rollout_mid — rollout to target at midpoint
        rm = rollout_mid[bi_arr, 0]  # (n_features,)
        result[:, 6] = rm

        # R8: rollout_ratio — final / mid
        result[:, 7] = to_target / (rm + 1e-10)

        return result.astype(np.float32)

    def _process_between_features_attention(
        self,
        attention_dict: dict[str, torch.Tensor | None],
        n_features: int,
        *,
        items_attention_dict: dict[str, torch.Tensor | None] | None = None,
    ) -> np.ndarray | None:
        """Process between-features attention into per-feature vectors.

        Attention shape from TabPFN: (batch_items, n_blocks, n_blocks, n_heads)
        where n_blocks = ceil(n_features / features_per_group) + 1 (last = target).

        Per feature × layer × head: 6 (or 15 if enriched) statistics.
        Plus 3 cross-layer summary stats.
        If enriched, also appends joint rollout stats (8 dims).
        """
        layer_attns = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None:
                if isinstance(attn, torch.Tensor):
                    a = attn.cpu().float()
                else:
                    a = torch.as_tensor(attn, dtype=torch.float32)
                a = torch.nan_to_num(a, nan=0.0)
                layer_attns.append(a)

        if not layer_attns:
            return None

        n_layers = len(layer_attns)
        n_blocks = layer_attns[0].shape[1]
        n_heads = layer_attns[0].shape[3]
        target_block = n_blocks - 1
        n_feature_blocks = n_blocks - 1

        # Map features to blocks — vectorized
        bi_arr = np.array([min(i * n_feature_blocks // n_features, n_feature_blocks - 1)
                           for i in range(n_features)])

        # Pre-compute mean over batch_items: list of numpy arrays (n_blocks, n_blocks, n_heads)
        mean_attns_np = [(attn.mean(dim=0)).numpy() for attn in layer_attns]

        # Pre-compute entropy: (n_layers, n_blocks, n_heads)
        entropy_all = np.zeros((n_layers, n_blocks, n_heads), dtype=np.float32)
        for li, a_np in enumerate(mean_attns_np):
            a_clamped = np.clip(a_np, 1e-10, None)
            entropy_all[li] = -(a_clamped * np.log(a_clamped)).sum(axis=1)

        # Compute per-layer per-head stats for all blocks at once
        n_stats_per_head = 15 if self.enriched else 6
        # block_stats: (n_layers, n_feature_blocks, n_heads, n_stats_per_head)
        block_stats = np.zeros((n_layers, n_feature_blocks, n_heads, n_stats_per_head), dtype=np.float32)

        for li, a_np in enumerate(mean_attns_np):
            for h in range(n_heads):
                ah = a_np[:, :, h]  # (n_blocks, n_blocks)
                feat_ah = ah[:n_feature_blocks, :]  # (n_feat_blocks, n_blocks)

                # 1. self-attention: diagonal of feature blocks
                block_stats[li, :, h, 0] = np.diag(ah)[:n_feature_blocks]
                # 2. to target
                block_stats[li, :, h, 1] = feat_ah[:, target_block]
                # 3. from target
                block_stats[li, :, h, 2] = ah[target_block, :n_feature_blocks]

                if n_feature_blocks > 1:
                    # For "mean to/from others" we need to exclude self for each block
                    # Outgoing: ah[bi, :n_fb] minus self → sum over others / (n_fb-1)
                    out_to_feats = feat_ah[:, :n_feature_blocks]  # (n_fb, n_fb)
                    in_from_feats = ah[:n_feature_blocks, :n_feature_blocks]  # same matrix transposed
                    row_sums = out_to_feats.sum(axis=1)  # (n_fb,)
                    diag_vals = np.diag(out_to_feats)
                    col_sums = out_to_feats.sum(axis=0)
                    n_others = max(n_feature_blocks - 1, 1)

                    # 4. mean to others = (row_sum - self) / n_others
                    block_stats[li, :, h, 3] = (row_sums - diag_vals) / n_others
                    # 5. mean from others = (col_sum - self) / n_others
                    block_stats[li, :, h, 4] = (col_sums - diag_vals) / n_others

                # 6. entropy
                block_stats[li, :, h, 5] = entropy_all[li, :n_feature_blocks, h]

                if self.enriched:
                    if n_feature_blocks > 1:
                        # For max/std to others, we need per-block masked stats
                        # Compute using sorted trick: for each row, sort and take stats of all but self
                        for stat_idx, bi_local in enumerate(range(n_feature_blocks)):
                            mask = np.ones(n_feature_blocks, dtype=bool)
                            mask[bi_local] = False
                            others_out = out_to_feats[bi_local][mask]
                            others_in = in_from_feats[:, bi_local][mask]

                            block_stats[li, bi_local, h, 6] = others_out.max()       # 7. max_to_others
                            block_stats[li, bi_local, h, 7] = others_out.std()        # 8. std_to_others
                            block_stats[li, bi_local, h, 8] = others_in.max()         # 9. max_from_others
                            block_stats[li, bi_local, h, 9] = others_in.std()         # 10. std_from_others

                            asym = out_to_feats[bi_local, :n_feature_blocks] - in_from_feats[:n_feature_blocks, bi_local]
                            asym_others = asym[mask]
                            block_stats[li, bi_local, h, 10] = asym_others.mean()     # 11. mean_asymmetry
                            block_stats[li, bi_local, h, 11] = np.abs(asym_others).max()  # 12. max_abs_asymmetry

                    # Target-relative ranking: vectorized
                    out_row = ah[:n_feature_blocks, :]   # (n_fb, n_blocks)
                    in_col = ah[:, :n_feature_blocks].T  # (n_fb, n_blocks)
                    to_target = feat_ah[:, target_block]     # (n_fb,)
                    from_target = ah[target_block, :n_feature_blocks]  # (n_fb,)

                    # 13. target_out_rank: fraction of blocks with attn <= to_target
                    block_stats[li, :, h, 12] = (out_row <= to_target[:, None]).mean(axis=1)
                    # 14. target_in_rank
                    block_stats[li, :, h, 13] = (in_col <= from_target[:, None]).mean(axis=1)
                    # 15. contrast_to_target = to_target - mean_to_others
                    block_stats[li, :, h, 14] = to_target - block_stats[li, :, h, 3]

        # Gather per-feature using block_indices
        # block_stats: (n_layers, n_feature_blocks, n_heads, n_stats)
        # gathered: (n_layers, n_features, n_heads, n_stats)
        gathered = block_stats[:, bi_arr, :, :]

        # Reshape: (n_features, n_layers * n_heads * n_stats)
        per_head_stats = gathered.transpose(1, 0, 2, 3).reshape(n_features, -1)

        # Cross-layer stats: mean attn-to-target per layer (mean over heads)
        # to_target_per_layer: (n_layers, n_feature_blocks)
        to_target_per_layer = np.stack([a[:n_feature_blocks, target_block, :].mean(axis=-1)
                                         for a in mean_attns_np])  # (n_layers, n_fb)
        # Gather per-feature: (n_layers, n_features)
        to_target_feat = to_target_per_layer[:, bi_arr]  # (n_layers, n_features)

        cross_layer = np.zeros((n_features, 3), dtype=np.float32)
        cross_layer[:, 0] = to_target_feat.mean(axis=0)
        cross_layer[:, 1] = to_target_feat.std(axis=0)
        if n_layers > 1:
            x_arr = np.arange(n_layers, dtype=np.float32)
            x_mean = x_arr.mean()
            x_var = ((x_arr - x_mean) ** 2).sum()
            if x_var > 0:
                cross_layer[:, 2] = ((x_arr[:, None] - x_mean) * (to_target_feat - to_target_feat.mean(axis=0))).sum(axis=0) / x_var

        result = np.concatenate([per_head_stats, cross_layer], axis=1)

        # Attention rollout (enriched mode only — needs both attention types)
        if self.enriched and items_attention_dict is not None:
            # Prepare items attention as averaged numpy arrays
            items_mean_np = []
            for key in sorted(items_attention_dict.keys()):
                ia = items_attention_dict[key]
                if ia is not None:
                    if isinstance(ia, torch.Tensor):
                        ia = ia.cpu().float()
                    else:
                        ia = torch.as_tensor(ia, dtype=torch.float32)
                    ia = torch.nan_to_num(ia, nan=0.0)
                    items_mean_np.append(ia.mean(dim=0).numpy())  # (n_items, n_items, n_heads)
                else:
                    items_mean_np.append(None)

            rollout_stats = self._compute_rollout_cpu(mean_attns_np, items_mean_np)
            if rollout_stats is not None:
                rollout_fv = self._gather_rollout_stats(
                    rollout_stats["rollout_final"],
                    rollout_stats["rollout_mid"],
                    n_features,
                    bi_arr,
                )
                result = np.concatenate([result, rollout_fv], axis=1)

        return result.astype(np.float32)

    @staticmethod
    def _compute_rollout_cpu(
        feat_mean_attns: list[np.ndarray],
        items_mean_attns: list[np.ndarray | None],
    ) -> dict[str, np.ndarray] | None:
        """Compute joint attention rollout on CPU.

        Combines both between-features and between-items attention into
        a joint N×N rollout matrix (N = n_blocks × n_items).

        Parameters
        ----------
        feat_mean_attns : list of (n_blocks, n_blocks, n_heads)
            Between-features attention, averaged over batch_items.
        items_mean_attns : list of (n_items, n_items, n_heads) or None
            Between-items attention, averaged over batch_blocks.

        Returns
        -------
        dict with 'rollout_final' (n_feat_blocks, n_blocks, 1) and
        'rollout_mid' (n_feat_blocks, 1), or None.
        """
        if not feat_mean_attns:
            return None

        n_layers = len(feat_mean_attns)
        n_blocks = feat_mean_attns[0].shape[0]
        n_feat_blocks = n_blocks - 1
        target_block = n_blocks - 1
        mid_layer = n_layers // 2

        # Determine n_items from items attention
        n_items = None
        for ia in items_mean_attns:
            if ia is not None:
                n_items = ia.shape[0]
                break

        if n_items is None:
            return None

        N = n_blocks * n_items
        I_N = np.eye(N, dtype=np.float32)
        rollout = I_N.copy()
        rollout_mid = None

        for li in range(n_layers):
            # Average over heads
            feat_2d = feat_mean_attns[li].mean(axis=-1)  # (n_blocks, n_blocks)
            feat_2d = np.nan_to_num(feat_2d, nan=0.0)

            items_a = items_mean_attns[li] if li < len(items_mean_attns) else None
            if items_a is not None:
                item_2d = items_a.mean(axis=-1)  # (n_items, n_items)
                item_2d = np.nan_to_num(item_2d, nan=0.0)
            else:
                item_2d = np.eye(n_items, dtype=np.float32)

            # Build N×N via Kronecker products
            J_items = np.ones((n_items, n_items), dtype=np.float32)
            J_blocks = np.ones((n_blocks, n_blocks), dtype=np.float32)

            A_feat = np.kron(feat_2d, J_items)   # (N, N)
            A_items = np.kron(J_blocks, item_2d)  # (N, N)

            # Add residual and normalize
            A_feat = A_feat + I_N
            A_feat = A_feat / (A_feat.sum(axis=-1, keepdims=True) + 1e-8)

            A_items = A_items + I_N
            A_items = A_items / (A_items.sum(axis=-1, keepdims=True) + 1e-8)

            # Sequential: items AFTER features
            A_layer = A_items @ A_feat
            rollout = A_layer @ rollout

            if li == mid_layer - 1:
                rollout_mid = _extract_block_scores_cpu(
                    rollout, n_blocks, n_items, n_feat_blocks, target_block
                )

        if rollout_mid is None:
            rollout_mid = _extract_block_scores_cpu(
                rollout, n_blocks, n_items, n_feat_blocks, target_block
            )

        # Extract final per-block-to-block rollout
        rollout_final = np.zeros((n_feat_blocks, n_blocks, 1), dtype=np.float32)
        for src_b in range(n_feat_blocks):
            src_start = src_b * n_items
            src_end = src_start + n_items
            for dst_b in range(n_blocks):
                dst_start = dst_b * n_items
                dst_end = dst_start + n_items
                rollout_final[src_b, dst_b, 0] = rollout[src_start:src_end, dst_start:dst_end].mean()

        return {
            "rollout_final": rollout_final,
            "rollout_mid": rollout_mid,
        }

    def _process_between_items_attention(
        self,
        attention_dict: dict[str, torch.Tensor | None],
        n_features: int,
        *,
        n_train: int = 0,
    ) -> np.ndarray | None:
        """Process between-items attention into per-feature summary statistics.

        Attention shape: (n_blocks, n_items, n_items, n_heads)
        where n_items = n_train + n_test.

        Extracts per-feature: entropy, max, variance (existing),
        plus train/test split stats: train→test, test→train,
        self-train, self-test, train-test ratio, concentration.
        """
        layer_attns = []
        for key in sorted(attention_dict.keys()):
            attn = attention_dict[key]
            if attn is not None:
                if isinstance(attn, torch.Tensor):
                    a = attn.cpu().float()
                    a = torch.nan_to_num(a, nan=0.0)
                else:
                    a = torch.as_tensor(attn, dtype=torch.float32)
                    a = torch.nan_to_num(a, nan=0.0)
                layer_attns.append(a)

        if not layer_attns:
            return None

        n_blocks_attn = layer_attns[0].shape[0]
        n_items = layer_attns[0].shape[1]
        n_heads = layer_attns[0].shape[3]

        bi_arr = np.array([min(i * n_blocks_attn // max(n_features, 1), n_blocks_attn - 1)
                           for i in range(n_features)])

        n_layers = len(layer_attns)
        has_split = n_train > 0 and n_train < n_items
        n_stats = 9 if self.enriched else 3

        # block_stats: (n_layers, n_blocks, n_heads, n_stats)
        block_stats = np.zeros((n_layers, n_blocks_attn, n_heads, n_stats), dtype=np.float32)

        for li, attn_tensor in enumerate(layer_attns):
            # attn_tensor: (n_blocks, n_items, n_items, n_heads) — torch CPU tensor
            # Process per-head to keep memory pressure manageable
            for h in range(n_heads):
                ah = attn_tensor[:, :, :, h]  # (n_blocks, n_items, n_items)

                # Entropy using fused kernel: entr(p) = -p*ln(p)
                row_ent = torch.special.entr(ah).sum(dim=2)  # (n_blocks, n_items)
                block_stats[li, :, h, 0] = row_ent.mean(dim=1).numpy()

                # Max and variance: flatten items dims
                ah_flat = ah.reshape(n_blocks_attn, -1)
                block_stats[li, :, h, 1] = ah_flat.max(dim=1).values.numpy()
                block_stats[li, :, h, 2] = ah_flat.var(dim=1).numpy()

                if self.enriched and has_split:
                    t2t = ah[:, :n_train, n_train:]
                    mean_t2t = t2t.mean(dim=(1, 2)).numpy()
                    mean_st = ah[:, :n_train, :n_train].mean(dim=(1, 2)).numpy()

                    block_stats[li, :, h, 3] = mean_t2t
                    block_stats[li, :, h, 4] = ah[:, n_train:, :n_train].mean(dim=(1, 2)).numpy()
                    block_stats[li, :, h, 5] = mean_st
                    block_stats[li, :, h, 6] = ah[:, n_train:, n_train:].mean(dim=(1, 2)).numpy()
                    block_stats[li, :, h, 7] = mean_t2t / (mean_st + 1e-10)
                    row_sums = ah.sum(dim=2).mean(dim=1).numpy()  # (n_blocks,)
                    block_stats[li, :, h, 8] = block_stats[li, :, h, 1] / (row_sums + 1e-10)

        # Gather per-feature: (n_layers, n_features, n_heads, n_stats)
        gathered = block_stats[:, bi_arr, :, :]
        return gathered.transpose(1, 0, 2, 3).reshape(n_features, -1).astype(np.float32)

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

        parts = []

        # --- Input gradients: vectorized over all features ---
        if input_grads is not None:
            grads_np = input_grads.cpu().float().numpy() if isinstance(input_grads, torch.Tensor) else np.asarray(input_grads, dtype=np.float32)
            if grads_np.ndim == 3:
                grads_np = grads_np[:, 0, :]  # (seq_len, n_features)

            n_grad_feats = min(grads_np.shape[1], n_features)
            abs_grads = np.abs(grads_np[:, :n_grad_feats])  # (seq_len, n_grad_feats)

            n_input_stats = 8 if self.enriched else 4
            input_stats = np.zeros((n_features, n_input_stats), dtype=np.float32)

            abs_mean = abs_grads.mean(axis=0)                     # (n_grad_feats,)
            input_stats[:n_grad_feats, 0] = abs_mean              # 1. abs_mean_grad
            input_stats[:n_grad_feats, 1] = abs_grads.max(axis=0) # 2. abs_max_grad
            input_stats[:n_grad_feats, 2] = grads_np[:, :n_grad_feats].std(axis=0)  # 3. grad_std
            pos_frac = (grads_np[:, :n_grad_feats] > 0).mean(axis=0)
            input_stats[:n_grad_feats, 3] = np.maximum(pos_frac, 1.0 - pos_frac)  # 4. dominance

            if self.enriched:
                feat_abs_means = abs_mean
                total_abs = feat_abs_means.sum()
                n_others = max(n_grad_feats - 1, 1)
                others_mean = (total_abs - feat_abs_means) / n_others

                # 5. rank
                ranks = np.zeros(n_grad_feats, dtype=np.float32)
                for i in range(n_grad_feats):
                    ranks[i] = (feat_abs_means <= feat_abs_means[i]).mean()
                input_stats[:n_grad_feats, 4] = ranks
                # 6. contrast
                input_stats[:n_grad_feats, 5] = abs_mean - others_mean
                # 7. abs_median
                input_stats[:n_grad_feats, 6] = np.median(abs_grads, axis=0)
                # 8. grad_energy
                input_stats[:n_grad_feats, 7] = (grads_np[:, :n_grad_feats] ** 2).mean(axis=0)

                # Fill defaults for features beyond gradient range
                if n_grad_feats < n_features:
                    input_stats[n_grad_feats:, 3] = 0.5  # dominance default
                    input_stats[n_grad_feats:, 4] = 0.5  # rank default
            else:
                if n_grad_feats < n_features:
                    input_stats[n_grad_feats:, 3] = 0.5

            parts.append(input_stats)

        # --- Attention gradients: vectorized ---
        if attn_grads is not None:
            sorted_keys = sorted(attn_grads.keys())
            n_attn_layers = len(sorted_keys)
            stats_per_head = 6 if self.enriched else 2

            # Pre-compute all layers to numpy at once
            layer_data = []
            for key in sorted_keys:
                grad = attn_grads[key]
                if grad is not None:
                    g_np = grad.cpu().float().numpy() if isinstance(grad, torch.Tensor) else np.asarray(grad, dtype=np.float32)
                    g_avg = g_np.mean(axis=0)  # (n_blocks, n_blocks, n_heads)
                    layer_data.append(g_avg)
                else:
                    layer_data.append(None)

            # Determine block mapping
            if layer_data[0] is not None:
                n_blocks = layer_data[0].shape[0]
                n_heads_g = layer_data[0].shape[2]
            else:
                n_heads_g = 3
                n_blocks = n_features + 1

            n_feat_blocks = n_blocks - 1
            target_block = n_blocks - 1
            bi_arr = np.array([min(i * n_feat_blocks // max(n_features, 1), n_feat_blocks - 1)
                               for i in range(n_features)])

            # block_attn_stats: (n_layers, n_feat_blocks, n_heads, stats_per_head)
            block_attn_stats = np.zeros((n_attn_layers, n_feat_blocks, n_heads_g, stats_per_head), dtype=np.float32)

            for li, g_avg in enumerate(layer_data):
                if g_avg is not None:
                    for h in range(n_heads_g):
                        gh = g_avg[:, :, h]  # (n_blocks, n_blocks)
                        block_attn_stats[li, :, h, 0] = gh[:n_feat_blocks, target_block]        # grad_to_target
                        block_attn_stats[li, :, h, 1] = np.abs(gh[:n_feat_blocks, :]).mean(axis=1)  # mean_abs_grad

                        if self.enriched:
                            block_attn_stats[li, :, h, 2] = gh[target_block, :n_feat_blocks]    # grad_from_target
                            block_attn_stats[li, :, h, 3] = gh[:n_feat_blocks, target_block] - gh[target_block, :n_feat_blocks]  # asymmetry
                            block_attn_stats[li, :, h, 4] = np.abs(gh[:n_feat_blocks, :]).max(axis=1)  # max_abs_grad
                            col_vals_abs = np.abs(gh[:, target_block])
                            for bi_local in range(n_feat_blocks):
                                block_attn_stats[li, bi_local, h, 5] = (col_vals_abs <= abs(gh[bi_local, target_block])).mean()

            # Gather per-feature
            gathered = block_attn_stats[:, bi_arr, :, :]  # (n_layers, n_features, n_heads, stats)
            attn_grad_flat = gathered.transpose(1, 0, 2, 3).reshape(n_features, -1)
            parts.append(attn_grad_flat)

            if self.enriched:
                # Cross-layer gradient summary (3)
                # grad_to_target mean over heads per layer: (n_layers, n_feat_blocks)
                g2t = block_attn_stats[:, :, :, 0].mean(axis=2)  # (n_layers, n_feat_blocks)
                g2t_abs = np.abs(g2t[:, bi_arr])  # (n_layers, n_features)

                cross = np.zeros((n_features, 3), dtype=np.float32)
                if n_attn_layers > 1:
                    x_arr = np.arange(n_attn_layers, dtype=np.float32)
                    x_mean = x_arr.mean()
                    x_var = ((x_arr - x_mean) ** 2).sum()
                    if x_var > 0:
                        cross[:, 0] = ((x_arr[:, None] - x_mean) * (g2t_abs - g2t_abs.mean(axis=0))).sum(axis=0) / x_var
                cross[:, 1] = g2t_abs.mean(axis=0)
                cross[:, 2] = g2t_abs.std(axis=0)
                parts.append(cross)

        if not parts:
            return None
        return np.concatenate(parts, axis=1).astype(np.float32)

    def _process_mlp_activations(
        self,
        activation_dict: dict[int, torch.Tensor],
        n_features: int,
    ) -> np.ndarray | None:
        """Process MLP activations into per-feature vectors.

        Activation shape: (batch, batch_items, n_blocks, emsize=192).
        Extracts per-block statistics plus relational stats (cosine to target,
        distinctiveness) and cross-layer summary.
        """
        if not activation_dict:
            return None

        first_act = list(activation_dict.values())[0]
        n_blocks = first_act.shape[2]
        n_feature_blocks = n_blocks - 1
        target_block = n_blocks - 1

        bi_arr = np.array([min(i * n_feature_blocks // max(n_features, 1), n_feature_blocks - 1)
                           for i in range(n_features)])

        # Pre-compute all layer activations averaged over batch/items
        sorted_layers = sorted(activation_dict.keys())
        n_layers = len(sorted_layers)
        # act_avgs: list of numpy (n_blocks, emsize)
        act_avgs = []
        for layer_idx in sorted_layers:
            act = activation_dict[layer_idx]
            if isinstance(act, torch.Tensor):
                act = act.cpu().float()
            act_avgs.append(act.mean(dim=(0, 1)).numpy())

        n_stats = 10 if self.enriched else 5
        # block_stats: (n_layers, n_feature_blocks, n_stats)
        block_stats = np.zeros((n_layers, n_feature_blocks, n_stats), dtype=np.float32)

        # Also track per-block norms for cross-layer: (n_layers, n_feature_blocks)
        norms_all = np.zeros((n_layers, n_feature_blocks), dtype=np.float32)

        for li, act_avg in enumerate(act_avgs):
            feat_acts = act_avg[:n_feature_blocks]  # (n_fb, emsize)
            target_act = act_avg[target_block]       # (emsize,)

            block_stats[li, :, 0] = feat_acts.mean(axis=1)                              # 1. mean
            block_stats[li, :, 1] = feat_acts.std(axis=1)                                # 2. std
            block_stats[li, :, 2] = feat_acts.max(axis=1)                                # 3. max
            block_stats[li, :, 3] = (np.abs(feat_acts) < 0.01).mean(axis=1)             # 4. sparsity
            norms = np.linalg.norm(feat_acts, axis=1)
            block_stats[li, :, 4] = norms                                                # 5. L2_norm
            norms_all[li] = norms

            if self.enriched:
                # 6. cosine_to_target: vectorized
                target_norm = np.linalg.norm(target_act) + 1e-10
                dot_products = feat_acts @ target_act
                block_stats[li, :, 5] = dot_products / (norms * target_norm + 1e-10)

                # 7. diff_from_mean
                all_blocks_mean = feat_acts.mean(axis=0)  # (emsize,)
                block_stats[li, :, 6] = np.linalg.norm(feat_acts - all_blocks_mean, axis=1)

                # 8. min
                block_stats[li, :, 7] = feat_acts.min(axis=1)

                # 9. skewness
                stds = block_stats[li, :, 1]
                means = block_stats[li, :, 0]
                centered = feat_acts - means[:, None]
                safe_stds = np.where(stds > 1e-10, stds, 1.0)
                block_stats[li, :, 8] = (centered ** 3).mean(axis=1) / (safe_stds ** 3)
                block_stats[li, :, 8] = np.where(stds > 1e-10, block_stats[li, :, 8], 0.0)

                # 10. pos_frac
                block_stats[li, :, 9] = (feat_acts > 0).mean(axis=1)

        # Gather per-feature: (n_layers, n_features, n_stats)
        gathered = block_stats[:, bi_arr, :]
        # Reshape: (n_features, n_layers * n_stats)
        per_layer = gathered.transpose(1, 0, 2).reshape(n_features, -1)

        if self.enriched:
            # Cross-layer summary (3 stats)
            norms_feat = norms_all[:, bi_arr]  # (n_layers, n_features)
            cross = np.zeros((n_features, 3), dtype=np.float32)

            if n_layers > 1:
                x_arr = np.arange(n_layers, dtype=np.float32)
                x_mean = x_arr.mean()
                x_var = ((x_arr - x_mean) ** 2).sum()
                if x_var > 0:
                    cross[:, 0] = ((x_arr[:, None] - x_mean) * (norms_feat - norms_feat.mean(axis=0))).sum(axis=0) / x_var

            # cosine_first_last
            first_acts = act_avgs[0][:n_feature_blocks]       # (n_fb, emsize)
            last_acts = act_avgs[-1][:n_feature_blocks]
            first_feat = first_acts[bi_arr]                    # (n_features, emsize)
            last_feat = last_acts[bi_arr]
            dot = (first_feat * last_feat).sum(axis=1)
            fn = np.linalg.norm(first_feat, axis=1) + 1e-10
            ln = np.linalg.norm(last_feat, axis=1) + 1e-10
            cross[:, 1] = dot / (fn * ln)

            # norm_ratio
            cross[:, 2] = ln / (fn + 1e-10)

            result = np.concatenate([per_layer, cross], axis=1)
        else:
            result = per_layer

        return result.astype(np.float32)

    def _process_items_attention_gradients(
        self,
        signals: dict[str, Any],
        n_features: int,
        *,
        n_train: int = 0,
    ) -> np.ndarray | None:
        """Process between-items attention gradients into per-feature vectors.

        Items attention gradient shape: (n_blocks, n_items, n_items, n_heads).
        Extracts per-feature grad stats with train/test split awareness.
        """
        items_grads = signals.get("items_attention_gradients")
        if items_grads is None:
            return None

        layer_grads = []
        for key in sorted(items_grads.keys()):
            grad = items_grads[key]
            if grad is not None:
                if isinstance(grad, torch.Tensor):
                    g = grad.cpu().float()
                else:
                    g = torch.as_tensor(grad, dtype=torch.float32)
                g = torch.nan_to_num(g, nan=0.0)
                layer_grads.append(g)

        if not layer_grads:
            return None

        n_blocks_grad = layer_grads[0].shape[0]
        n_items = layer_grads[0].shape[1]
        n_heads = layer_grads[0].shape[3]

        bi_arr = np.array([min(i * n_blocks_grad // max(n_features, 1), n_blocks_grad - 1)
                           for i in range(n_features)])

        n_layers = len(layer_grads)
        has_split = n_train > 0 and n_train < n_items
        n_stats = 6

        # Vectorized block stats: (n_layers, n_blocks, n_heads, n_stats)
        block_grad_stats = np.zeros((n_layers, n_blocks_grad, n_heads, n_stats), dtype=np.float32)

        for li, grad_tensor in enumerate(layer_grads):
            g_np = grad_tensor.numpy()  # (n_blocks, n_items, n_items, n_heads)
            abs_g = np.abs(g_np)

            if has_split:
                block_grad_stats[li, :, :, 0] = abs_g[:, :n_train, n_train:, :].mean(axis=(1, 2))  # grad_train_to_test
                block_grad_stats[li, :, :, 1] = abs_g[:, n_train:, :n_train, :].mean(axis=(1, 2))  # grad_test_to_train
                block_grad_stats[li, :, :, 2] = abs_g[:, :n_train, :n_train, :].mean(axis=(1, 2))  # grad_self_train

            block_grad_stats[li, :, :, 3] = abs_g.mean(axis=(1, 2))  # grad_mean_abs
            block_grad_stats[li, :, :, 4] = abs_g.max(axis=(1, 2))   # grad_max_abs
            mean_val = block_grad_stats[li, :, :, 3]
            block_grad_stats[li, :, :, 5] = block_grad_stats[li, :, :, 4] / (mean_val + 1e-10)  # concentration

        # Gather per-feature
        gathered = block_grad_stats[:, bi_arr, :, :]  # (n_layers, n_features, n_heads, n_stats)
        return gathered.transpose(1, 0, 2, 3).reshape(n_features, -1).astype(np.float32)

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
