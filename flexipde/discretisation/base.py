"""
Base classes for discretisation schemes.

The :class:`BaseDifferentiator` defines the minimal interface for
discretisation methods used by flexipde.  All differentiators take a
grid as input and provide methods to compute gradients, divergences and
Laplacians of scalar or vector fields.

By adhering to this interface, different schemes (spectral, finite
difference, finite element) can be swapped at runtime without
modifying the high‑level model or solver code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Any


class BaseDifferentiator(ABC):
    """Abstract base class for discretisation schemes."""

    grid: Any

    @abstractmethod
    def grad(self, u: Any, axis: int) -> Any:
        """Compute the gradient of a scalar field along a given axis."""
        raise NotImplementedError

    @abstractmethod
    def divergence(self, vec: Sequence[Any]) -> Any:
        """Compute the divergence of a vector field."""
        raise NotImplementedError

    @abstractmethod
    def laplacian(self, u: Any) -> Any:
        """Compute the Laplacian of a scalar field."""
        raise NotImplementedError
