"""
Finite difference differentiation.

This module implements simple finite difference stencils on uniform
grids.  It supports periodic and non‑periodic boundaries.  The
finite difference method is second‑order accurate by default; higher
order schemes may be added in the future.

This differentiator is suitable for problems with arbitrary boundary
conditions, in contrast to the spectral method which assumes
periodicity.  For example, diffusion equations with fixed
temperature at the boundaries or plasma transport in a bounded domain
can be solved using finite differences.

The design is inspired by the modularity of BOUT++, which allows a
variety of numerical methods to be swapped at runtime【501252340464299†L24-L33】.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

from .base import BaseDifferentiator


@dataclass
class FiniteDifference(BaseDifferentiator):
    """Finite difference differentiator.

    Parameters
    ----------
    grid : :class:`~flexipde.grid.Grid`
        The grid on which differentiation will be performed.  Spacing must
        be uniform within each dimension.
    order : int, optional
        The order of the finite difference scheme.  Currently only
        ``2`` (second order) is supported.
    backend : str, optional
        Either ``"jax"`` or ``"numpy"``.  If JAX is unavailable, the
        numpy backend is used.
    """

    grid: Any
    order: int = 2
    backend: str | None = None

    def __post_init__(self) -> None:
        if self.order != 2:
            raise NotImplementedError("Only second‑order finite differences are supported for now")
        if self.backend is None:
            self.backend = "jax" if _HAS_JAX else "numpy"
        if self.backend == "jax" and not _HAS_JAX:
            raise RuntimeError("JAX backend requested but JAX is not available")
        # Convert grid to requested backend
        if self.backend == "jax":
            self.grid = self.grid.to_jax()
        # Precompute spacings as arrays (could be scalar or array depending on coordinate spacing)
        self._dx = []
        # grid.spacing returns tuple of arrays representing spacing at first interval; treat as constant
        for d in self.grid.spacing():
            # spacing may be a scalar or array; convert to float
            if _np.isscalar(d):
                self._dx.append(d)
            else:
                # take mean value
                self._dx.append(float(_np.asarray(d).mean()))

    def _roll(self, u: Any, shift: int, axis: int) -> Any:
        if self.backend == "jax":
            return jnp.roll(u, shift, axis)
        return _np.roll(u, shift, axis)

    def grad(self, u: Any, axis: int) -> Any:
        """Compute ∂u/∂x_axis using second‑order finite differences."""
        dx = self._dx[axis]
        # central difference for interior points
        forward = self._roll(u, -1, axis)
        backward = self._roll(u, 1, axis)
        deriv = (forward - backward) / (2.0 * dx)
        # fix boundaries for non‑periodic dimensions
        if not self.grid.periodic[axis]:
            # first point: forward difference
            slicer = [slice(None)] * u.ndim
            slicer[axis] = 0
            leading = u[tuple(slicer)]
            slicer[axis] = 1
            nextp = u[tuple(slicer)]
            deriv = deriv.copy() if self.backend == "numpy" else deriv.at[slicer].set((nextp - leading) / dx)
            # last point: backward difference
            slicer[axis] = -1
            trailing = u[tuple(slicer)]
            slicer[axis] = -2
            prevp = u[tuple(slicer)]
            last_idx = [-1]  # placeholder to update last index along axis
            # Build tuple to index
            idx = [slice(None)] * u.ndim
            idx[axis] = -1
            if self.backend == "numpy":
                deriv[tuple(idx)] = (trailing - prevp) / dx
            else:
                deriv = deriv.at[tuple(idx)].set((trailing - prevp) / dx)
        return deriv

    def divergence(self, vec: Sequence[Any]) -> Any:
        if len(vec) != self.grid.dim:
            raise ValueError(f"Expected {self.grid.dim} components, got {len(vec)}")
        res = None
        for axis, comp in enumerate(vec):
            dcomp = self.grad(comp, axis)
            res = dcomp if res is None else res + dcomp
        return res

    def laplacian(self, u: Any) -> Any:
        result = None
        for axis in range(self.grid.dim):
            dx = self._dx[axis]
            forward = self._roll(u, -1, axis)
            backward = self._roll(u, 1, axis)
            second = (forward - 2.0 * u + backward) / (dx * dx)
            if not self.grid.periodic[axis]:
                # forward/backward difference at boundaries
                # first point: second derivative using two forward points
                slicer0 = [slice(None)] * u.ndim
                slicer0[axis] = 0
                u0 = u[tuple(slicer0)]
                slicer1 = slicer0.copy(); slicer1[axis] = 1
                u1 = u[tuple(slicer1)]
                slicer2 = slicer0.copy(); slicer2[axis] = 2
                u2 = u[tuple(slicer2)] if u.shape[axis] > 2 else u1
                if self.backend == "numpy":
                    second[tuple(slicer0)] = (u2 - 2*u1 + u0) / (dx * dx)
                else:
                    second = second.at[tuple(slicer0)].set((u2 - 2*u1 + u0) / (dx*dx))
                # last point: two backward points
                slicer_end = [slice(None)] * u.ndim
                slicer_end[axis] = -1
                un1 = u[tuple(slicer_end)]
                slicer3 = slicer_end.copy(); slicer3[axis] = -2
                un2 = u[tuple(slicer3)]
                slicer4 = slicer_end.copy(); slicer4[axis] = -3 if u.shape[axis] > 2 else -2
                un3 = u[tuple(slicer4)] if u.shape[axis] > 2 else un2
                idx_last = [slice(None)] * u.ndim; idx_last[axis] = -1
                if self.backend == "numpy":
                    second[tuple(idx_last)] = (un3 - 2*un2 + un1) / (dx*dx)
                else:
                    second = second.at[tuple(idx_last)].set((un3 - 2*un2 + un1) / (dx*dx))
            result = second if result is None else result + second
        return result
