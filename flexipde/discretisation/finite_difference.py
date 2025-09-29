# mypy: ignore-errors
"""Finite difference discretisation.

This module implements simple central finite difference stencils for
derivatives on structured grids.  It supports both periodic and
non‑periodic boundaries.  For non‑periodic dimensions, a one‑sided
difference is used at the boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence

import numpy as _np

try:
    import jax.numpy as _jnp  # type: ignore[attr-defined]
    _JAX_AVAILABLE = True
except Exception:
    _jnp = None
    _JAX_AVAILABLE = False

from ..grid import Grid


@dataclass
class FiniteDifference:
    """Finite difference derivative operator.

    Parameters
    ----------
    grid:
        The spatial grid.
    backend:
        Either ``"numpy"`` or ``"jax"``.  If JAX is requested but not
        available, NumPy will be used instead.
    """

    grid: Grid
    backend: str = "numpy"

    def __post_init__(self) -> None:
        if self.backend == "jax" and _JAX_AVAILABLE:
            self._xp = _jnp
        else:
            self._xp = _np
        # Precompute spacings
        self._dx = self.grid.spacing()

    def grad(self, u: Any) -> List[Any]:
        xp = self._xp
        grads = []
        for axis, (dx, per) in enumerate(zip(self._dx, self.grid.periodic)):
            # roll for periodic; otherwise compute one‑sided at boundaries
            if per:
                forward = xp.roll(u, -1, axis=axis)
                backward = xp.roll(u, 1, axis=axis)
                du = (forward - backward) / (2.0 * dx)
            else:
                forward = xp.roll(u, -1, axis=axis)
                backward = xp.roll(u, 1, axis=axis)
                du = (forward - backward) / (2.0 * dx)
                # correct boundaries using one‑sided difference
                # forward difference at first point
                slicer = [slice(None)] * u.ndim
                slicer[axis] = 0
                idx = tuple(slicer)
                slicer2 = slicer.copy()
                slicer2[axis] = 1
                idx2 = tuple(slicer2)
                du = du.copy()
                du[idx] = (u[idx2] - u[idx]) / dx
                # backward difference at last point
                slicer[axis] = -1
                idx = tuple(slicer)
                slicer2[axis] = -2
                idx2 = tuple(slicer2)
                du[idx] = (u[idx] - u[idx2]) / dx
            grads.append(du)
        return grads

    def divergence(self, vec: Sequence[Any]) -> Any:
        xp = self._xp
        if len(vec) != self.grid.ndim:
            raise ValueError("vector field must have one component per dimension")
        result = xp.zeros_like(vec[0])
        for axis, comp in enumerate(vec):
            dcomp = self.grad(comp)[axis]
            result = result + dcomp
        return result

    def laplacian(self, u: Any) -> Any:
        xp = self._xp
        lap = xp.zeros_like(u)
        for axis, (dx, per) in enumerate(zip(self._dx, self.grid.periodic)):
            if per:
                forward = xp.roll(u, -1, axis=axis)
                backward = xp.roll(u, 1, axis=axis)
                lap = lap + (forward - 2.0 * u + backward) / (dx * dx)
            else:
                forward = xp.roll(u, -1, axis=axis)
                backward = xp.roll(u, 1, axis=axis)
                lap_axis = (forward - 2.0 * u + backward) / (dx * dx)
                # correct boundaries with one‑sided second derivative
                sl0 = [slice(None)] * u.ndim
                sl0[axis] = 0
                idx0 = tuple(sl0)
                sl1 = sl0.copy()
                sl1[axis] = 1
                idx1 = tuple(sl1)
                sl2 = sl0.copy()
                sl2[axis] = 2 if u.shape[axis] > 2 else 1
                idx2 = tuple(sl2)
                lap_axis = lap_axis.copy()
                lap_axis[idx0] = (u[idx2] - 2.0 * u[idx1] + u[idx0]) / (dx * dx)
                # last point
                sl0[axis] = -1
                idx0 = tuple(sl0)
                sl1[axis] = -2
                idx1 = tuple(sl1)
                sl2[axis] = -3 if u.shape[axis] > 2 else -2
                idx2 = tuple(sl2)
                lap_axis[idx0] = (u[idx0] - 2.0 * u[idx1] + u[idx2]) / (dx * dx)
                lap = lap + lap_axis
        return lap