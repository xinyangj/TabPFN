"""Extract all internal signals from TabPFN inference.

Extracts attention weights, embeddings, gradients, and MLP activations
from a TabPFN model during inference, to be used as input features
for the interpretation model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn


class SignalExtractor:
    """Extract comprehensive internal signals from TabPFN inference.

    Collects 5 categories of signals:
    1. Between-features attention weights (12 layers × 6 heads)
    2. Between-items attention weights (12 layers × 6 heads)
    3. Encoder embeddings (train + test, 192-dim)
    4. Gradients (input, attention, embedding)
    5. MLP activations (per layer)

    Examples
    --------
    >>> from tabpfn import TabPFNRegressor
    >>> regressor = TabPFNRegressor(n_estimators=1)
    >>> regressor.fit(X_train, y_train)
    >>> extractor = SignalExtractor()
    >>> signals = extractor.extract(regressor, X_train, y_train, X_test)
    """

    def __init__(self, *, extract_gradients: bool = True) -> None:
        """Initialize the signal extractor.

        Parameters
        ----------
        extract_gradients : bool
            Whether to extract gradient signals. Requires backward pass.
        """
        self.extract_gradients = extract_gradients

    def extract(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> dict[str, Any]:
        """Extract all internal signals from a TabPFN inference pass.

        Parameters
        ----------
        model : TabPFNRegressor
            A fitted TabPFN regressor model.
        X_train : np.ndarray
            Training features, shape (n_train, n_features).
        y_train : np.ndarray
            Training targets, shape (n_train,).
        X_test : np.ndarray
            Test features, shape (n_test, n_features).

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - "between_features_attention": dict of per-layer attention weight tensors
            - "between_items_attention": dict of per-layer attention weight tensors
            - "train_embeddings": tensor of shape (n_train, emsize)
            - "test_embeddings": tensor of shape (n_test, emsize)
            - "input_gradients": tensor of shape (n_samples, n_features) or None
            - "attention_gradients": dict of per-layer gradient tensors or None
            - "mlp_activations": dict of per-layer activation tensors
            - "n_features": int
            - "n_train": int
            - "n_test": int
        """
        architecture = self._get_architecture(model)
        encoder_layers = list(architecture.transformer_encoder.layers)
        n_layers = len(encoder_layers)

        # Enable attention weight collection
        self._enable_attention_extraction(encoder_layers, grad=self.extract_gradients)

        # Register hooks for MLP activations
        mlp_activations: dict[int, torch.Tensor] = {}
        hooks = self._register_mlp_hooks(encoder_layers, mlp_activations)

        try:
            # Prepare input data
            X_combined, y_combined, single_eval_pos = self._prepare_inputs(
                model, X_train, y_train, X_test
            )

            # Run forward pass
            if self.extract_gradients:
                signals = self._forward_with_gradients(
                    model,
                    architecture,
                    X_combined,
                    y_combined,
                    single_eval_pos,
                    encoder_layers,
                    n_layers,
                    mlp_activations,
                )
            else:
                signals = self._forward_without_gradients(
                    model,
                    architecture,
                    X_combined,
                    y_combined,
                    single_eval_pos,
                    encoder_layers,
                    n_layers,
                    mlp_activations,
                )

            signals["n_features"] = X_train.shape[1]
            signals["n_train"] = X_train.shape[0]
            signals["n_test"] = X_test.shape[0]

            return signals

        finally:
            # Clean up hooks and disable attention collection
            for hook in hooks:
                hook.remove()
            self._disable_attention_extraction(encoder_layers)

    def _get_architecture(self, model: Any) -> nn.Module:
        """Get the underlying architecture from a TabPFN model."""
        # Access the model's architecture
        if hasattr(model, "models_") and model.models_:
            return model.models_[0]
        elif hasattr(model, "model"):
            return model.model
        raise ValueError("Cannot find TabPFN architecture in the model.")

    def _enable_attention_extraction(
        self, layers: list[nn.Module], *, grad: bool = False
    ) -> None:
        """Enable attention weight extraction on all layers."""
        for layer in layers:
            if layer.self_attn_between_features is not None:
                layer.self_attn_between_features.enable_attention_weights_return(True)
                if grad:
                    layer.self_attn_between_features.enable_attention_grad_retention(
                        True
                    )
            layer.self_attn_between_items.enable_attention_weights_return(True)

    def _disable_attention_extraction(self, layers: list[nn.Module]) -> None:
        """Disable attention weight extraction on all layers."""
        for layer in layers:
            if layer.self_attn_between_features is not None:
                layer.self_attn_between_features.enable_attention_weights_return(False)
                layer.self_attn_between_features.enable_attention_grad_retention(False)
            layer.self_attn_between_items.enable_attention_weights_return(False)

    def _register_mlp_hooks(
        self, layers: list[nn.Module], storage: dict[int, torch.Tensor]
    ) -> list[torch.utils.hooks.RemovableHook]:
        """Register forward hooks to capture MLP activations."""
        hooks = []
        for i, layer in enumerate(layers):

            def hook_fn(
                module: nn.Module,
                input: tuple[torch.Tensor, ...],
                output: torch.Tensor,
                layer_idx: int = i,
            ) -> None:
                storage[layer_idx] = output.detach()

            hooks.append(layer.mlp.register_forward_hook(hook_fn))
        return hooks

    def _prepare_inputs(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare and preprocess inputs for the forward pass.

        Returns combined X, y tensors and single_eval_pos.
        """
        device = next(self._get_architecture(model).parameters()).device
        dtype = next(self._get_architecture(model).parameters()).dtype

        X_all = np.vstack([X_train, X_test])
        x_tensor = torch.from_numpy(X_all.astype(np.float32)).to(device=device)

        # Add sequence and batch dims: (seq_len, batch=1, n_features)
        x_tensor = x_tensor.unsqueeze(1)

        y_tensor = torch.from_numpy(y_train.astype(np.float32)).to(device=device)
        y_tensor = y_tensor.unsqueeze(1)  # (n_train, batch=1)

        single_eval_pos = len(X_train)

        return x_tensor, y_tensor, single_eval_pos

    def _forward_without_gradients(
        self,
        model: Any,
        architecture: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        single_eval_pos: int,
        encoder_layers: list[nn.Module],
        n_layers: int,
        mlp_activations: dict[int, torch.Tensor],
    ) -> dict[str, Any]:
        """Run forward pass without gradient computation."""
        with torch.no_grad():
            output = architecture.forward(
                x, y, only_return_standard_out=False
            )

        return self._collect_signals(
            encoder_layers,
            n_layers,
            output,
            mlp_activations,
            input_gradients=None,
            attention_gradients=None,
        )

    def _forward_with_gradients(
        self,
        model: Any,
        architecture: nn.Module,
        x: torch.Tensor,
        y: torch.Tensor,
        single_eval_pos: int,
        encoder_layers: list[nn.Module],
        n_layers: int,
        mlp_activations: dict[int, torch.Tensor],
    ) -> dict[str, Any]:
        """Run forward pass with gradient computation."""
        x_input = x.detach().clone().requires_grad_(True)

        output = architecture.forward(
            x_input, y, only_return_standard_out=False
        )

        # Compute gradients: use mean of test predictions as scalar loss
        predictions = output["standard"]
        loss = predictions.mean()
        loss.backward(retain_graph=True)

        # Collect attention gradients
        attention_gradients: dict[str, torch.Tensor | None] = {}
        for i, layer in enumerate(encoder_layers):
            if layer.self_attn_between_features is not None:
                attn_w = layer.self_attn_between_features.get_attention_weights()
                attention_gradients[f"layer_{i}"] = (
                    attn_w.grad.detach().clone() if attn_w is not None and attn_w.grad is not None else None
                )
            else:
                attention_gradients[f"layer_{i}"] = None

        input_gradients = x_input.grad.detach().clone() if x_input.grad is not None else None

        return self._collect_signals(
            encoder_layers,
            n_layers,
            {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in output.items()},
            mlp_activations,
            input_gradients=input_gradients,
            attention_gradients=attention_gradients,
        )

    def _collect_signals(
        self,
        encoder_layers: list[nn.Module],
        n_layers: int,
        output: dict[str, Any],
        mlp_activations: dict[int, torch.Tensor],
        *,
        input_gradients: torch.Tensor | None,
        attention_gradients: dict[str, torch.Tensor | None] | None,
    ) -> dict[str, Any]:
        """Collect all extracted signals into a dictionary."""
        between_features_attention: dict[str, torch.Tensor | None] = {}
        between_items_attention: dict[str, torch.Tensor | None] = {}

        for i, layer in enumerate(encoder_layers):
            if layer.self_attn_between_features is not None:
                attn = layer.self_attn_between_features.get_attention_weights()
                between_features_attention[f"layer_{i}"] = (
                    attn.detach().clone() if attn is not None else None
                )
            else:
                between_features_attention[f"layer_{i}"] = None

            attn_items = layer.self_attn_between_items.get_attention_weights()
            between_items_attention[f"layer_{i}"] = (
                attn_items.detach().clone() if attn_items is not None else None
            )

        # Extract embeddings
        train_emb = output.get("train_embeddings")
        test_emb = output.get("test_embeddings")

        return {
            "between_features_attention": between_features_attention,
            "between_items_attention": between_items_attention,
            "train_embeddings": train_emb.detach().clone() if isinstance(train_emb, torch.Tensor) else None,
            "test_embeddings": test_emb.detach().clone() if isinstance(test_emb, torch.Tensor) else None,
            "input_gradients": input_gradients,
            "attention_gradients": attention_gradients,
            "mlp_activations": {k: v.clone() for k, v in mlp_activations.items()},
        }
