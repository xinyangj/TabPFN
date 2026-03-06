"""Loss functions for the interpretation model.

Supports different loss functions for different label modes:
- BCE for binary modes (direct parents, ancestry)
- MSE for continuous modes (graded ancestry, interventional)
- Combined multi-task loss
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn


class InterpretationLoss(nn.Module):
    """Loss function for the interpretation model.

    Parameters
    ----------
    mode : str
        Loss mode corresponding to the label type:
        - "binary": Binary cross-entropy (for binary_direct, binary_ancestry)
        - "continuous": Mean squared error (for graded_ancestry, interventional)
        - "multi_task": Combined loss across multiple label modes
    pos_weight : float, optional
        Positive class weight for binary mode (handles class imbalance,
        since most features are NOT causal parents). Default 3.0.
    """

    def __init__(
        self,
        mode: Literal["binary", "continuous", "multi_task"] = "binary",
        *,
        pos_weight: float = 3.0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]),
        )
        self.mse = nn.MSELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss.

        Parameters
        ----------
        predictions : torch.Tensor
            Model predictions, shape (batch, n_features) — raw logits for binary,
            scores for continuous.
        targets : torch.Tensor
            Ground truth, shape (batch, n_features).
        mask : torch.Tensor, optional
            Boolean mask for valid features (1=valid, 0=padding), shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        if mask is not None:
            predictions = predictions[mask]
            targets = targets[mask]

        if self.mode == "binary":
            return self.bce(predictions, targets)
        elif self.mode == "continuous":
            return self.mse(predictions, targets)
        elif self.mode == "multi_task":
            # Assumes predictions has 2× features: first half binary, second half continuous
            mid = predictions.shape[-1] // 2
            loss_binary = self.bce(predictions[..., :mid], targets[..., :mid])
            loss_continuous = self.mse(predictions[..., mid:], targets[..., mid:])
            return loss_binary + loss_continuous
        else:
            raise ValueError(f"Unknown loss mode: {self.mode}")
