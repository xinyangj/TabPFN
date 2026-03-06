"""PyTorch Dataset for interpretation model training.

Stores pairs of (extracted_signals, ground_truth_importance) for training
the interpretation model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class InterpretationDataset(Dataset):
    """Dataset of (signal_features, importance_labels) pairs.

    Each item is a single synthetic dataset that has been processed through
    TabPFN to extract signals.

    Parameters
    ----------
    data : list[dict]
        List of processed dataset records, each containing:
        - "feature_vectors": np.ndarray of shape (n_features, D_total)
        - "labels": dict mapping label mode names to np.ndarray of shape (n_features,)
    label_mode : str
        Which label mode to use: "binary_direct", "binary_ancestry",
        "graded_ancestry", or "interventional".
    max_features : int
        Maximum number of features (for padding). Datasets with more features
        are truncated, those with fewer are zero-padded.
    augment : bool
        Whether to apply data augmentation (feature permutation).
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        *,
        label_mode: str = "binary_direct",
        max_features: int = 50,
        augment: bool = True,
    ) -> None:
        self.data = data
        self.label_mode = label_mode
        self.max_features = max_features
        self.augment = augment

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        record = self.data[idx]
        feature_vectors = record["feature_vectors"]  # (n_features, D_total)
        labels = record["labels"][self.label_mode]  # (n_features,)

        n_features, d_total = feature_vectors.shape

        # Optional: random feature permutation for augmentation
        if self.augment and self.training_mode:
            perm = np.random.permutation(n_features)
            feature_vectors = feature_vectors[perm]
            labels = labels[perm]

        # Pad or truncate to max_features
        if n_features > self.max_features:
            feature_vectors = feature_vectors[: self.max_features]
            labels = labels[: self.max_features]
            mask = np.ones(self.max_features, dtype=np.float32)
        else:
            pad_size = self.max_features - n_features
            feature_vectors = np.pad(
                feature_vectors, ((0, pad_size), (0, 0)), mode="constant"
            )
            labels = np.pad(labels, (0, pad_size), mode="constant")
            mask = np.zeros(self.max_features, dtype=np.float32)
            mask[:n_features] = 1.0

        return {
            "features": torch.from_numpy(feature_vectors.astype(np.float32)),
            "labels": torch.from_numpy(labels.astype(np.float32)),
            "mask": torch.from_numpy(mask),
            "n_features": n_features,
        }

    @property
    def training_mode(self) -> bool:
        """Check if augmentation should be applied."""
        return self.augment

    @property
    def feature_dim(self) -> int:
        """Get the feature dimension D_total."""
        if self.data:
            return self.data[0]["feature_vectors"].shape[1]
        return 0

    @classmethod
    def from_disk(
        cls,
        path: str | Path,
        **kwargs: Any,
    ) -> InterpretationDataset:
        """Load a dataset from disk.

        Parameters
        ----------
        path : str or Path
            Path to a .npz file containing the dataset.
        **kwargs
            Additional arguments passed to the constructor.
        """
        path = Path(path)
        loaded = np.load(path, allow_pickle=True)
        data = loaded["data"].tolist()
        return cls(data, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save dataset to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=np.array(self.data, dtype=object))
