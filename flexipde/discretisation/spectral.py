"""Fourier spectral discretisation.

This module provides :class:`SpectralDifferentiator` for computing
derivatives using the Fourier transform.  It supports both NumPy and JAX
backends.  Spectral methods achieve machine‑precision accuracy for smooth
functions on periodic domains.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import numpy as _np

try:
    import jax.numpy as _jnp  # type: ignore[attr-defined]
    import jax as _jax  # type: ignore[attr-defined]
    _JAX_AVAILABLE = True
except Exception:  # pragma: no cover
    _jax = None
    _jnp = None
    _JAX_AVAILABLE = False

from ..grid import Grid


@dataclass
class SpectralDifferentiator:
    """Compute derivatives using Fourier transforms.

    Parameters
    ----------
    grid:
        The spatial grid.  All dimensions must be uniformly spaced.
    backend:
        Either ``"numpy"`` or ``"jax"``.  When set to ``"jax```` and
        JAX is installed, operations are performed with JAX arrays and JIT
        compiled; otherwise NumPy is used.  If JAX is requested but not
        installed, NumPy will be used and a warning emitted.
    """

    grid: Grid
    backend: str = "numpy"

    def __post_init__(self) -> None:
        # Choose array library
        if self.backend == "jax" and _JAX_AVAILABLE:
            xp = _jnp
        else:
            xp = _np
        self._xp = xp
        self._backend = "jax" if xp is _jnp else "numpy"

        # Precompute wavenumber arrays for each dimension
        self._k_axes: List[Any] = []
        spacings = self.grid.spacing()
        for n, dx in zip(self.grid.shape, spacings):
            # wavenumbers: 0,1,...,N/2-1, -N/2,...,-1 multiplied by 2π/L
            # Use fftfreq on the full domain length
            k = xp.fft.fftfreq(n, d=dx) * 2.0 * xp.pi
            self._k_axes.append(k)

        # FFT and inverse FFT functions
        # Use numpy or jax numpy accordingly
        if self._backend == "jax":
            self._fft = _jnp.fft.fftn  # type: ignore[assignment]
            self._ifft = _jnp.fft.ifftn  # type: ignore[assignment]
        else:
            self._fft = _np.fft.fftn  # type: ignore[assignment]
            self._ifft = _np.fft.ifftn  # type: ignore[assignment]

    def grad(self, u: Any) -> List[Any]:
        """Compute the gradient of a scalar field.

        Parameters
        ----------
        u:
            An array representing the scalar field on the grid.  The shape
            must match ``grid.shape``.

        Returns
        -------
        list of arrays
            The gradient components along each dimension.
        """
        xp = self._xp
        u_hat = self._fft(u)
        grads: List[Any] = []
        # Compute derivative along each axis by multiplying by 1j*k
        for axis, k in enumerate(self._k_axes):
            # Reshape k to broadcast along all dimensions
            shape = [1] * u_hat.ndim
            shape[axis] = k.size
            k_reshaped = k.reshape(shape)
            deriv_hat = (1j * k_reshaped) * u_hat
            du = self._ifft(deriv_hat)
            grads.append(xp.real(du))
        return grads

    def divergence(self, vec: Sequence[Any]) -> Any:
        """Compute the divergence of a vector field.

        Parameters
        ----------
        vec:
            A sequence of arrays, one per dimension, representing the vector
            field components.  Each array must have shape ``grid.shape``.

        Returns
        -------
        array
            The divergence scalar field.
        """
        xp = self._xp
        if len(vec) != self.grid.ndim:
            raise ValueError("vector field must have one component per dimension")
        result_hat = None
        for axis, (comp, k) in enumerate(zip(vec, self._k_axes)):
            comp_hat = self._fft(comp)
            # Reshape k for broadcasting
            shape = [1] * comp_hat.ndim
            shape[axis] = k.size
            k_reshaped = k.reshape(shape)
            part_hat = (1j * k_reshaped) * comp_hat
            if result_hat is None:
                result_hat = part_hat
            else:
                result_hat = result_hat + part_hat
        div = self._ifft(result_hat)
        return xp.real(div)

    def laplacian(self, u: Any) -> Any:
        """Compute the Laplacian of a scalar field.

        Parameters
        ----------
        u:
            A scalar field array of shape ``grid.shape``.

        Returns
        -------
        array
            The Laplacian of ``u``.
        """
        xp = self._xp
        u_hat = self._fft(u)
        # Build k^2 sum term
        k_sq = 0
        for axis, k in enumerate(self._k_axes):
            shape = [1] * u_hat.ndim
            shape[axis] = k.size
            k_reshaped = k.reshape(shape)
            k_sq = k_sq + (k_reshaped * k_reshaped)
        lap_hat = -k_sq * u_hat
        result = self._ifft(lap_hat)
        return xp.real(result)