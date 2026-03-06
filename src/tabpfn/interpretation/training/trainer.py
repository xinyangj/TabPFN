"""Training loop for the interpretation model."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tabpfn.interpretation.model.interpretation_model import InterpretationModel
from tabpfn.interpretation.model.losses import InterpretationLoss
from tabpfn.interpretation.training.dataset import InterpretationDataset

logger = logging.getLogger(__name__)


class Trainer:
    """Training loop for the interpretation model.

    Parameters
    ----------
    model : InterpretationModel
        The model to train.
    loss_fn : InterpretationLoss
        Loss function.
    lr : float
        Learning rate.
    weight_decay : float
        Weight decay for AdamW.
    device : str
        Training device.

    Examples
    --------
    >>> model = InterpretationModel("mlp", input_dim=512)
    >>> trainer = Trainer(model)
    >>> trainer.train(train_dataset, val_dataset, n_epochs=50)
    """

    def __init__(
        self,
        model: InterpretationModel,
        *,
        loss_fn: InterpretationLoss | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)

        if loss_fn is None:
            loss_mode = "binary" if model.output_mode == "binary" else "continuous"
            loss_fn = InterpretationLoss(mode=loss_mode)
        self.loss_fn = loss_fn.to(self.device)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )

    def train(
        self,
        train_dataset: InterpretationDataset,
        val_dataset: InterpretationDataset | None = None,
        *,
        n_epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
        save_path: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Train the interpretation model.

        Parameters
        ----------
        train_dataset : InterpretationDataset
            Training data.
        val_dataset : InterpretationDataset, optional
            Validation data for early stopping.
        n_epochs : int
            Maximum number of epochs.
        batch_size : int
            Batch size.
        patience : int
            Early stopping patience (epochs without improvement).
        save_path : str or Path, optional
            Path to save the best model checkpoint.

        Returns
        -------
        dict[str, list[float]]
            Training history with "train_loss", "val_loss", etc.
        """
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            if val_dataset
            else None
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(n_epochs):
            # Training
            train_loss = self._train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            # Validation
            val_loss = float("inf")
            if val_loader:
                val_loss = self._eval_epoch(val_loader)
                history["val_loss"].append(val_loss)

            self.scheduler.step()

            logger.info(
                f"Epoch {epoch + 1}/{n_epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
            )

            # Early stopping
            if val_loader and val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                if save_path:
                    self.model.save(save_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save final model if no validation
        if save_path and val_loader is None:
            self.model.save(save_path)

        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(features, mask=mask)
            loss = self.loss_fn(predictions, labels, mask=mask.bool())
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        """Run a single evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            predictions = self.model(features, mask=mask)
            loss = self.loss_fn(predictions, labels, mask=mask.bool())

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
