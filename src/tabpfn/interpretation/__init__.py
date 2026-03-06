"""Post-hoc interpretation module for TabPFN feature importance.

This module trains a model to predict the causal importance of each tabular
feature based on TabPFN's internal representations (attention weights,
embeddings, gradients, and activations).

The interpretation model is trained on synthetic data generated from Structural
Causal Models (SCMs) where the ground-truth causal structure is known.
"""

from __future__ import annotations

from tabpfn.interpretation.extraction.signal_extractor import SignalExtractor
from tabpfn.interpretation.extraction.signal_processor import SignalProcessor
from tabpfn.interpretation.model.interpretation_model import InterpretationModel
from tabpfn.interpretation.synthetic_data.scm_generator import SCMGenerator

__all__ = [
    "InterpretationModel",
    "SCMGenerator",
    "SignalExtractor",
    "SignalProcessor",
]
