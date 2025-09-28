"""
Discretisation schemes for flexipde.

This package provides pluggable numerical differentiation routines.  Each
scheme encapsulates the logic for computing spatial derivatives on a
given :class:`~flexipde.grid.Grid`.  Users can select a scheme via the
configuration file or by passing the appropriate differentiator class
directly when constructing a model.

Two schemes are provided in the initial release:

* :mod:`flexipde.discretisation.spectral` – spectral (Fourier) methods
  suitable for periodic problems.  Spectral methods provide high
  accuracy and minimal numerical dissipation, making them well suited
  for linear advection and wave problems.  They leverage the Fast
  Fourier Transform (FFT) implemented in JAX or NumPy.
* :mod:`flexipde.discretisation.finite_difference` – finite difference
  stencils for first and second derivatives.  These methods work on
  both periodic and non‑periodic domains and are useful for problems
  with boundary conditions like Dirichlet or Neumann.

In the future the library may include finite element or finite volume
methods.  The modular design means that adding a new scheme simply
requires implementing the same public API: initialise with a
:class:`Grid` and provide functions ``grad``, ``div`` and
``laplacian``.  See the base classes for guidance.
"""

from .spectral import SpectralDifferentiator
from .finite_difference import FiniteDifference

__all__ = ["SpectralDifferentiator", "FiniteDifference"]