"""Spatial discretisation schemes.

The :mod:`flexipde.discretisation` subpackage contains classes that compute
spatial derivatives on structured grids.  Two schemes are provided out of
the box:

* :class:`~flexipde.discretisation.spectral.SpectralDifferentiator` uses
  Fourier transforms to compute derivatives exactly on periodic domains.
* :class:`~flexipde.discretisation.finite_difference.FiniteDifference` uses
  finite difference stencils to approximate derivatives on non‑periodic
  domains.

Each discretiser exposes methods ``grad``, ``divergence`` and
``laplacian``.  The ``backend`` argument controls whether the
implementation uses NumPy or JAX; if JAX is available, JIT compilation
provides significant speedups.
"""

from .spectral import SpectralDifferentiator
from .finite_difference import FiniteDifference

__all__ = [
    "SpectralDifferentiator",
    "FiniteDifference",
]