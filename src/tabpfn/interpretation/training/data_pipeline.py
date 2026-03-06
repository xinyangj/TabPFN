"""End-to-end data generation pipeline.

Generates synthetic datasets, runs TabPFN inference, extracts signals,
computes ground-truth labels, and assembles training data for the
interpretation model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
from tabpfn.interpretation.synthetic_data.label_generator import (
    LabelMode,
    compute_all_labels,
)
from tabpfn.interpretation.synthetic_data.scm_generator import SCMGenerator

logger = logging.getLogger(__name__)


class DataPipeline:
    """End-to-end pipeline for generating interpretation model training data.

    Steps:
    1. Generate synthetic dataset from SCM
    2. Fit TabPFN and extract internal signals
    3. Process signals into per-feature vectors
    4. Compute ground-truth importance labels
    5. Return (feature_vectors, labels) pair

    Parameters
    ----------
    scm_generator : SCMGenerator
        Generator for synthetic datasets.
    extract_gradients : bool
        Whether to extract gradient signals (slower but more informative).
    device : str
        Device for TabPFN inference.

    Examples
    --------
    >>> pipeline = DataPipeline(scm_generator=SCMGenerator(seed=42))
    >>> data = pipeline.generate_batch(n_datasets=100)
    >>> print(len(data), data[0]["feature_vectors"].shape)
    """

    def __init__(
        self,
        *,
        scm_generator: SCMGenerator | None = None,
        extract_gradients: bool = True,
        device: str = "auto",
    ) -> None:
        if scm_generator is None:
            scm_generator = SCMGenerator()
        self.scm_generator = scm_generator
        self.signal_extractor = SignalExtractor(extract_gradients=extract_gradients)
        self.signal_processor = SignalProcessor()
        self.device = device

    def generate_single(
        self,
        *,
        n_features: int | None = None,
        n_samples: int | None = None,
    ) -> dict[str, Any]:
        """Generate a single training example.

        Returns
        -------
        dict
            Contains "feature_vectors" (np.ndarray) and "labels" (dict).
        """
        from tabpfn import TabPFNRegressor

        # Step 1: Generate synthetic dataset
        dataset = self.scm_generator.generate(
            n_features=n_features, n_samples=n_samples
        )
        logger.info(
            f"Generated dataset: {dataset.X.shape}, "
            f"target parents: {list(dataset.dag.predecessors(dataset.target_node))}"
        )

        # Step 2: Split into train/test
        n_total = dataset.X.shape[0]
        n_train = max(int(n_total * 0.7), 10)
        X_train, X_test = dataset.X[:n_train], dataset.X[n_train:]
        y_train, y_test = dataset.y[:n_train], dataset.y[n_train:]

        # Step 3: Fit TabPFN and extract signals
        regressor = TabPFNRegressor(n_estimators=1, device=self.device)
        regressor.fit(X_train, y_train)

        signals = self.signal_extractor.extract(
            regressor, X_train, y_train, X_test
        )

        # Step 4: Process signals into per-feature vectors
        feature_vectors = self.signal_processor.process(signals)

        # Step 5: Compute ground-truth labels
        labels = compute_all_labels(dataset)

        return {
            "feature_vectors": feature_vectors,
            "labels": labels,
            "metadata": dataset.metadata,
        }

    def generate_batch(
        self,
        n_datasets: int,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generate a batch of training examples.

        Parameters
        ----------
        n_datasets : int
            Number of datasets to generate.
        **kwargs
            Additional arguments passed to generate_single().

        Returns
        -------
        list[dict]
            List of training examples.
        """
        results = []
        for i in range(n_datasets):
            try:
                result = self.generate_single(**kwargs)
                results.append(result)
                logger.info(f"Generated dataset {i + 1}/{n_datasets}")
            except Exception as e:
                logger.warning(f"Failed to generate dataset {i + 1}: {e}")
                continue
        return results
