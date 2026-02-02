"""Attention extraction utilities for Gene Regulatory Network (GRN) inference.

This module provides functionality to extract attention weights from TabPFN models
for the purpose of inferring gene regulatory networks. The attention mechanism
in TabPFN captures relationships between transcription factors (TFs) and target genes.

Key insight: TabPFN uses dual attention mechanisms:
1. Between-features attention: Captures TF-TF relationships
2. Between-items attention: Captures sample-wise patterns

For GRN inference, we're primarily interested in the between-features attention,
as it directly indicates regulatory relationships between TFs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    from collections.abc import MutableMapping


class AttentionExtractor:
    """Extract attention weights from TabPFN models.

    This class extracts attention weights from TabPFN models by enabling
    the attention weight return functionality in the MultiHeadAttention modules.

    Examples
    --------
    >>> from tabpfn import TabPFNRegressor
    >>> from tabpfn.grn import AttentionExtractor
    >>> model = TabPFNRegressor(n_estimators=1)
    >>> model.fit(X_train, y_train)
    >>> extractor = AttentionExtractor()
    >>> attention_weights = extractor.extract(model, X_test)
    >>> print(attention_weights.keys())
    dict_keys(['layer_0', 'layer_1', ...])
    """

    def __init__(self) -> None:
        """Initialize the AttentionExtractor."""
        pass

    def extract(
        self,
        model: Any,
        X: torch.Tensor | list,
        *,
        max_layers: int | None = None,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Extract attention weights from a fitted TabPFN model.

        Parameters
        ----------
        model : Any
            Fitted TabPFN model (TabPFNRegressor or TabPFNClassifier)
        X : torch.Tensor or list
            Input data to extract attention for
        max_layers : int, optional
            Maximum number of layers to extract attention from.
            If None, extracts from all layers.

        Returns
        -------
        attention_weights : dict
            Nested dictionary containing attention weights:
            {
                'layer_0': {
                    'between_features': Tensor of shape (batch, seqlen_q, seqlen_k, nheads),
                    'between_items': Tensor of shape (batch, seqlen_q, seqlen_k, nheads)
                },
                'layer_1': {...},
                ...
            }
        """

        # Get the underlying model architectures
        if not hasattr(model, "models_") or not model.models_:
            raise ValueError("Model must be fitted before extracting attention")

        # Enable attention weights return on all attention modules
        # and collect references to the modules
        attn_modules: dict[str, Any] = {}
        for model_idx, model_arch in enumerate(model.models_):
            if hasattr(model_arch, "transformer_encoder"):
                encoder = model_arch.transformer_encoder
                n_layers = min(max_layers, len(encoder.layers)) if max_layers else len(
                    encoder.layers
                )

                for layer_idx in range(n_layers):
                    layer = encoder.layers[layer_idx]

                    # Enable attention weights return and store reference
                    if hasattr(layer, "self_attn_between_features") and layer.self_attn_between_features is not None:
                        layer.self_attn_between_features.enable_attention_weights_return(True)
                        attn_modules[f"layer_{layer_idx}_between_features"] = layer.self_attn_between_features

                    if hasattr(layer, "self_attn_between_items") and layer.self_attn_between_items is not None:
                        layer.self_attn_between_items.enable_attention_weights_return(True)
                        attn_modules[f"layer_{layer_idx}_between_items"] = layer.self_attn_between_items

        # Run forward pass to capture attention weights
        with torch.no_grad():
            _ = model.predict(X)

        # Retrieve attention weights from modules BEFORE disabling
        attention_weights = self._retrieve_attention_weights(attn_modules)

        # Disable attention weights return after retrieval
        for module in attn_modules.values():
            module.enable_attention_weights_return(False)

        return attention_weights

    def _retrieve_attention_weights(
        self,
        attn_modules: dict[str, Any],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Retrieve attention weights from attention modules.

        Parameters
        ----------
        attn_modules : dict
            Dictionary mapping layer/attention type to module references

        Returns
        -------
        attention_weights : dict
            Dictionary of attention weights organized by layer and type
        """
        attention_weights: dict[str, dict[str, torch.Tensor]] = {}

        for key, module in attn_modules.items():
            # Parse key like "layer_0_between_features"
            parts = key.split("_")
            layer_idx = parts[1]
            attn_type = "_".join(parts[2:])  # "between_features" or "between_items"

            layer_key = f"layer_{layer_idx}"
            if layer_key not in attention_weights:
                attention_weights[layer_key] = {}

            # Get attention weights from module
            weights = module.get_attention_weights()
            if weights is not None:
                attention_weights[layer_key][attn_type] = weights

        return attention_weights



class EdgeScoreComputer:
    """Compute edge scores from attention weights for GRN inference.

    This class takes attention weights extracted from TabPFN and computes
    edge scores that represent the strength of regulatory relationships
    between transcription factors (TFs) and target genes.

    Parameters
    ----------
    aggregation_method : str, default='mean'
        Method to aggregate attention across heads and layers.
        Options: 'mean', 'max', 'last_layer'

    Examples
    --------
    >>> from tabpfn.grn import EdgeScoreComputer
    >>> computer = EdgeScoreComputer(aggregation_method='mean')
    >>> edge_scores = computer.compute(attention_weights)
    >>> print(edge_scores.shape)  # (n_tfs, n_targets)
    """

    def __init__(self, aggregation_method: str = "mean") -> None:
        """Initialize the EdgeScoreComputer.

        Parameters
        ----------
        aggregation_method : str, default='mean'
            Method to aggregate attention across heads and layers
        """
        if aggregation_method not in ["mean", "max", "last_layer"]:
            raise ValueError(
                f"aggregation_method must be 'mean', 'max', or 'last_layer', "
                f"got '{aggregation_method}'"
            )
        self.aggregation_method = aggregation_method

    def compute(
        self,
        attention_weights: dict[str, dict[str, torch.Tensor]],
        *,
        use_between_features: bool = True,
        use_between_items: bool = False,
    ) -> torch.Tensor:
        """Compute edge scores from attention weights.

        Parameters
        ----------
        attention_weights : dict
            Nested dictionary of attention weights from AttentionExtractor
        use_between_features : bool, default=True
            Whether to use between-features attention
        use_between_items : bool, default=False
            Whether to use between-items attention

        Returns
        -------
        edge_scores : torch.Tensor
            Edge score matrix of shape (n_tfs, n_targets)
        """
        if not use_between_features and not use_between_items:
            raise ValueError(
                "At least one of use_between_features or use_between_items must be True"
            )

        attention_patterns = []

        for layer_key in sorted(attention_weights.keys()):
            layer_data = attention_weights[layer_key]

            if use_between_features and "between_features" in layer_data:
                attn = layer_data["between_features"]
                attention_patterns.append(attn)

            if use_between_items and "between_items" in layer_data:
                attn = layer_data["between_items"]
                attention_patterns.append(attn)

        if not attention_patterns:
            raise ValueError("No attention patterns found in attention_weights")

        # Aggregate attention patterns
        if self.aggregation_method == "last_layer":
            aggregated = attention_patterns[-1]
        elif self.aggregation_method == "mean":
            aggregated = torch.stack(attention_patterns).mean(dim=0)
        elif self.aggregation_method == "max":
            aggregated = torch.stack(attention_patterns).max(dim=0)[0]
        else:
            raise ValueError(f"Unknown aggregation_method: {self.aggregation_method}")

        # The aggregated tensor might be multi-dimensional
        # We need to extract the relevant dimensions for TF-target edges
        # For now, return the aggregated attention as-is
        # In a full implementation, we'd extract the TF-target specific scores

        return aggregated
