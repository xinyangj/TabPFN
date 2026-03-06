"""Edge functions for Structural Causal Models.

Each edge function defines how a parent node influences a child node
in the causal graph. These are sampled randomly during synthetic data
generation to produce diverse causal relationships.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import nn


class EdgeFunction:
    """Base class for edge functions in an SCM."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearEdge(EdgeFunction):
    """Linear relationship: y += w * x + b."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.weight = rng.standard_normal() * 2.0
        self.bias = rng.standard_normal() * 0.3

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.weight * x + self.bias


class PolynomialEdge(EdgeFunction):
    """Polynomial relationship: y += w1*x + w2*x^2 + w3*x^3."""

    def __init__(self, rng: np.random.Generator, degree: int = 2) -> None:
        self.coefficients = rng.standard_normal(degree + 1) * 1.5
        self.coefficients[0] *= 0.3  # smaller bias

    def __call__(self, x: np.ndarray) -> np.ndarray:
        result = np.zeros_like(x)
        for i, coef in enumerate(self.coefficients):
            result += coef * (x**i)
        return result


class MLPEdge(EdgeFunction):
    """Small MLP relationship: y += MLP(x)."""

    def __init__(self, rng: np.random.Generator, hidden_dim: int = 16) -> None:
        # Initialize a small random MLP
        seed = int(rng.integers(0, 2**31))
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Scale output
        with torch.no_grad():
            self.net[-1].weight.mul_(2.0)

    @torch.no_grad()
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(-1)
        out = self.net(x_t).squeeze(-1).numpy()
        return out


class SigmoidEdge(EdgeFunction):
    """Sigmoidal relationship: y += a * sigmoid(b * x + c)."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.a = rng.standard_normal() * 3.0
        self.b = rng.standard_normal() * 2.0
        self.c = rng.standard_normal() * 0.5

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.a / (1.0 + np.exp(-(self.b * x + self.c)))


class InteractionEdge(EdgeFunction):
    """Interaction term: applies to two parent inputs combined.

    y += w * x1 * x2 (when called with concatenated parent signals).
    """

    def __init__(self, rng: np.random.Generator) -> None:
        self.weight = rng.standard_normal() * 1.5

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.weight * x


class NoiseOnlyEdge(EdgeFunction):
    """No real relationship — just additive noise.

    Used to create "distractor" edges that have no causal effect.
    """

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)


EdgeType = Literal["linear", "polynomial", "mlp", "sigmoid", "noise"]

EDGE_TYPE_PROBS: dict[EdgeType, float] = {
    "linear": 0.35,
    "polynomial": 0.25,
    "mlp": 0.20,
    "sigmoid": 0.15,
    "noise": 0.05,
}


def sample_edge_function(
    rng: np.random.Generator,
    edge_type: EdgeType | None = None,
) -> EdgeFunction:
    """Sample a random edge function.

    Parameters
    ----------
    rng : np.random.Generator
        Random number generator.
    edge_type : str, optional
        If specified, create this specific edge type. Otherwise, sample
        randomly from the distribution in EDGE_TYPE_PROBS.

    Returns
    -------
    EdgeFunction
        A callable edge function.
    """
    if edge_type is None:
        types = list(EDGE_TYPE_PROBS.keys())
        probs = list(EDGE_TYPE_PROBS.values())
        edge_type = rng.choice(types, p=probs)

    if edge_type == "linear":
        return LinearEdge(rng)
    elif edge_type == "polynomial":
        return PolynomialEdge(rng)
    elif edge_type == "mlp":
        return MLPEdge(rng)
    elif edge_type == "sigmoid":
        return SigmoidEdge(rng)
    elif edge_type == "noise":
        return NoiseOnlyEdge()
    else:
        raise ValueError(f"Unknown edge type: {edge_type}")
