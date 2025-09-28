"""
Spectral differentiation.

This module implements Fourier spectral differentiation on uniform
periodic grids.  Given a :class:`~flexipde.grid.Grid` with periodic
boundary conditions, the :class:`SpectralDifferentiator` computes
derivatives by transforming fields to Fourier space, multiplying by
appropriate wavenumber factors and transforming back.  This yields
spectral (infinite order) accuracy for smooth periodic functions.

Spectral methods are ideal for problems where periodicity can be
assumed.  The high accuracy and low dissipation make them popular in
plasma physics, for example in simulating Alfvén waves or turbulence.

References
----------
Φ‑Flow uses Fourier transforms to provide reusable simulation code that
works for both 2D and 3D fluids and supports multiple backends
(NumPy, PyTorch, TensorFlow, JAX)【383486560010699†L364-L376】.  flexipde adopts a similar
approach for spectral differentiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Sequence, Any

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    from jax.numpy.fft import fftn as jfftn, ifftn as jifftn, rfftn as jrfftn, irfftn as jirfftn
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

from .base import BaseDifferentiator

@dataclass
class SpectralDifferentiator(BaseDifferentiator):
    """Fourier spectral differentiator.

    Parameters
    ----------
    grid : :class:`~flexipde.grid.Grid`
        The grid on which differentiation will be performed.  All
        dimensions must be periodic.  If the grid is not periodic, the
        differentiator will raise a ``ValueError`` during initialisation.
    backend : str, optional
        Either ``"jax"`` or ``"numpy"``.  If ``"jax"`` is requested but JAX is
        not available, a ``RuntimeError`` is raised.  If ``None``, the
        backend is automatically selected based on availability of JAX.
    """

    grid: Any
    backend: str | None = None

    def __post_init__(self) -> None:
        # Check that the grid is periodic in all dimensions
        if not all(self.grid.periodic):
            raise ValueError("SpectralDifferentiator requires all dimensions to be periodic")
        # Choose backend
        if self.backend is None:
            self.backend = "jax" if _HAS_JAX else "numpy"
        if self.backend == "jax" and not _HAS_JAX:
            raise RuntimeError("JAX backend requested but JAX is not available")
        # Convert grid to correct backend
        if self.backend == "jax":
            self.grid = self.grid.to_jax()
        # Precompute wavenumber arrays for each dimension.  Each entry is a 1D
        # array of complex wavenumbers for that axis.  We deliberately
        # annotate the list element type as ``Any`` to avoid mypy errors
        # when mixing NumPy and JAX array types (which are not compatible at
        # type-check time).  Using ``Any`` prevents the type checker from
        # inferring a single concrete type for all entries.  See tests for
        # coverage.
        self._k_arrays: list[Any] = []
        shape = self.grid.shape
        for axis, n in enumerate(shape):
            # Determine length of domain in this dimension from coordinate array
            coords = self.grid.coordinates[axis]
            # coordinate difference along axis; assume uniform spacing
            # Use numpy for difference; jnp if jax
            arr0 = _np.asarray(coords.take(0, axis))
            arr1 = _np.asarray(coords.take(1, axis))
            dx = _np.abs(arr1 - arr0)
            length = dx * n
            # compute wavenumbers: k = 2π * i * freq.  Use distinct variable
            # names to avoid shadowing across different scopes (mypy no-redef).
            if self.backend == "jax":
                freqs = jnp.fft.fftfreq(n, d=float(dx))
                # Compute spectral wavenumbers (ik).  Use a distinct
                # variable name per axis to avoid mypy complaining about
                # redefinition across the loop.
                kaxis: Any = 2 * jnp.pi * (1j) * freqs  # type: ignore[assignment]
                self._k_arrays.append(kaxis)
            else:
                freqs = _np.fft.fftfreq(n, d=float(dx))
                kaxis_np: Any = 2 * _np.pi * 1j * freqs  # type: ignore[assignment]
                self._k_arrays.append(kaxis_np)

    def _fft(self, u: Any) -> Any:
        """Compute the forward Fourier transform of a field."""
        if self.backend == "jax":
            return jfftn(u)
        return _np.fft.fftn(u)

    def _ifft(self, u_hat: Any) -> Any:
        """Compute the inverse Fourier transform of a field."""
        if self.backend == "jax":
            return jifftn(u_hat)
        return _np.fft.ifftn(u_hat)

    def grad(self, u: Any, axis: int) -> Any:
        """Compute the partial derivative ∂u/∂x_axis.

        Parameters
        ----------
        u : array
            Input scalar field.
        axis : int
            Dimension along which to differentiate.
        Returns
        -------
        array
            The derivative of the field along the specified axis.
        """
        u_hat = self._fft(u)
        # Multiply by ik along axis; broadcast across other dims
        # Retrieve the wavenumber array for this axis.  We copy into a
        # local variable with a distinct name to avoid redefinition warnings.
        k_axis = self._k_arrays[axis]
        # reshape k for broadcasting to same shape as u_hat
        # Add singleton dimensions for other axes
        shape = [1] * u_hat.ndim
        shape[axis] = k_axis.shape[0]
        k_broad = k_axis.reshape(shape)
        du_hat = k_broad * u_hat
        du = self._ifft(du_hat)
        # Real part is derivative (imag part should be negligible)
        if self.backend == "jax":
            return jnp.real(du)
        return _np.real(du)

    def divergence(self, vec: Sequence[Any]) -> Any:
        """Compute divergence of a vector field.

        Parameters
        ----------
        vec : sequence of arrays
            A sequence of length ``dim`` containing the field components.
        Returns
        -------
        array
            The divergence ∇·vec.
        """
        if len(vec) != self.grid.dim:
            raise ValueError(f"Expected {self.grid.dim} components, got {len(vec)}")
        # Sum derivatives along each axis
        res = None
        for axis, comp in enumerate(vec):
            dcomp = self.grad(comp, axis)
            res = dcomp if res is None else res + dcomp
        return res

    def laplacian(self, u: Any) -> Any:
        """Compute the Laplacian Δu = ∑_i ∂²u/∂x_i²."""
        u_hat = self._fft(u)
        # Sum squared wavenumbers across axes
        # Compute (-k^2) factor (note: derivative is multiplication by (ik);
        # second derivative multiplies by -(k^2))
        # Build k^2 grid by broadcasting
        k2_sum = None
        for axis, k_axis in enumerate(self._k_arrays):
            # reshape k for broadcasting
            shape = [1] * u_hat.ndim
            shape[axis] = k_axis.shape[0]
            kgrid = k_axis.reshape(shape)
            # negative squared: -(k^2)
            term = -(kgrid * kgrid)
            k2_sum = term if k2_sum is None else k2_sum + term
        du_hat = k2_sum * u_hat
        du = self._ifft(du_hat)
        if self.backend == "jax":
            return jnp.real(du)
        return _np.real(du)
