"""Interpretation model architectures.

Provides two model variants:
1. PerFeatureMLP: Independent MLP applied to each feature (permutation equivariant)
2. PerFeatureTransformer: Self-attention across features + per-feature output
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn


class PerFeatureMLP(nn.Module):
    """Per-feature MLP for predicting feature importance.

    The same MLP is applied independently to each feature's signal vector,
    making the model naturally permutation equivariant.

    Parameters
    ----------
    input_dim : int
        Dimension of the per-feature signal vector (D_total).
    hidden_dims : list[int]
        Hidden layer dimensions.
    dropout : float
        Dropout probability.
    output_mode : str
        "binary" for sigmoid output, "continuous" for raw output.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.1,
        output_mode: Literal["binary", "continuous"] = "binary",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.output_mode = output_mode

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Per-feature signal vectors, shape (batch, n_features, D_total).
        mask : torch.Tensor, optional
            Validity mask, shape (batch, n_features). 1=valid, 0=padding.

        Returns
        -------
        torch.Tensor
            Per-feature importance scores, shape (batch, n_features).
            Raw logits for binary mode, scores for continuous mode.
        """
        # Apply same MLP to each feature independently
        # x: (B, F, D) -> (B*F, D) -> MLP -> (B*F, 1) -> (B, F)
        B, F, D = x.shape
        out = self.mlp(x.reshape(B * F, D))  # (B*F, 1)
        out = out.reshape(B, F)  # (B, F)

        if mask is not None:
            out = out * mask.float()

        return out


class PerFeatureTransformer(nn.Module):
    """Transformer-based model for predicting feature importance.

    Treats each feature's signal vector as a token, applies self-attention
    across features, then outputs per-feature importance scores.

    Parameters
    ----------
    input_dim : int
        Dimension of the per-feature signal vector.
    d_model : int
        Transformer hidden dimension.
    n_heads : int
        Number of attention heads.
    n_layers : int
        Number of transformer layers.
    dropout : float
        Dropout probability.
    output_mode : str
        "binary" or "continuous".
    """

    def __init__(
        self,
        input_dim: int,
        *,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_mode: Literal["binary", "continuous"] = "binary",
    ) -> None:
        super().__init__()
        self.output_mode = output_mode

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Per-feature signal vectors, shape (batch, n_features, D_total).
        mask : torch.Tensor, optional
            Validity mask, shape (batch, n_features). 1=valid, 0=padding.

        Returns
        -------
        torch.Tensor
            Per-feature importance scores, shape (batch, n_features).
        """
        # Project to d_model
        h = self.input_proj(x)  # (B, F, d_model)

        # Create key_padding_mask for transformer (True = ignore)
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = ~mask.bool()

        # Self-attention across features
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)  # (B, F, d_model)

        # Per-feature output
        out = self.output_head(h).squeeze(-1)  # (B, F)

        if mask is not None:
            out = out * mask.float()

        return out


class InterpretationModel(nn.Module):
    """Wrapper combining model variant selection with loading/saving.

    Parameters
    ----------
    variant : str
        Model variant: "mlp" or "transformer".
    input_dim : int
        Dimension of per-feature signal vector.
    output_mode : str
        "binary" or "continuous".
    **kwargs
        Additional arguments passed to the model variant.
    """

    def __init__(
        self,
        variant: Literal["mlp", "transformer"] = "mlp",
        input_dim: int = 512,
        *,
        output_mode: Literal["binary", "continuous"] = "binary",
        **kwargs: int | float | list[int],
    ) -> None:
        super().__init__()
        self.variant = variant
        self.input_dim = input_dim
        self.output_mode = output_mode

        if variant == "mlp":
            self.model = PerFeatureMLP(
                input_dim, output_mode=output_mode, **kwargs
            )
        elif variant == "transformer":
            self.model = PerFeatureTransformer(
                input_dim, output_mode=output_mode, **kwargs
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass — predict per-feature importance.

        Parameters
        ----------
        x : torch.Tensor
            Per-feature signal vectors, shape (batch, n_features, D_total).
        mask : torch.Tensor, optional
            Validity mask, shape (batch, n_features).

        Returns
        -------
        torch.Tensor
            Per-feature importance scores, shape (batch, n_features).
        """
        return self.model(x, mask=mask)

    def predict(self, feature_vectors: torch.Tensor) -> torch.Tensor:
        """Predict feature importance scores.

        Parameters
        ----------
        feature_vectors : torch.Tensor
            Per-feature signal vectors, shape (n_features, D_total).

        Returns
        -------
        torch.Tensor
            Importance scores, shape (n_features,).
        """
        self.eval()
        with torch.no_grad():
            x = feature_vectors.unsqueeze(0)  # Add batch dim
            logits = self.forward(x)
            if self.output_mode == "binary":
                return torch.sigmoid(logits).squeeze(0)
            return logits.squeeze(0)

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "variant": self.variant,
                "input_dim": self.input_dim,
                "output_mode": self.output_mode,
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def from_pretrained(cls, path: str | Path) -> InterpretationModel:
        """Load a pretrained model from disk.

        Parameters
        ----------
        path : str or Path
            Path to the saved model checkpoint.

        Returns
        -------
        InterpretationModel
            Loaded model ready for inference.
        """
        checkpoint = torch.load(path, weights_only=False)
        model = cls(
            variant=checkpoint["variant"],
            input_dim=checkpoint["input_dim"],
            output_mode=checkpoint["output_mode"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
