r"""Drift–kinetic equation (simplified).

The drift–kinetic model describes the evolution of a distribution function
``f(x, v, t)`` in phase space.  Here we implement a minimal 1D version
without self‑consistency: the particles advect in space with velocity ``v``
and are accelerated in velocity space by a constant electric field ``E``.
The equation is

.. math::

    \partial_t f + v \partial_x f + E \partial_v f = 0.

"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as _np

from ..grid import Grid
from ..discretisation import SpectralDifferentiator, FiniteDifference
from .base import PDEModel


@dataclass
class DriftKinetic(PDEModel):
    """A simplified drift–kinetic model.

    Parameters
    ----------
    grid:
        The spatial grid (1D).
    diff:
        The spatial discretiser for the spatial derivative.
    nv:
        Number of velocity grid points.
    v_min, v_max:
        Range of the velocity grid.  Velocity points are uniformly spaced.
    E:
        Constant electric field accelerating particles in velocity space.
    """

    nv: int = 32
    v_min: float = -5.0
    v_max: float = 5.0
    E: float = 0.0

    def __post_init__(self) -> None:
        # velocity grid for phase space derivative
        self.v_grid = _np.linspace(self.v_min, self.v_max, self.nv)
        self.dv = (self.v_max - self.v_min) / (self.nv - 1)
        self.field_names = ["f"]
        super().__post_init__()

    def initial_state(self, ic_params: dict[str, Any] | None = None) -> dict[str, Any]:
        # f is a 2D array (x, v)
        ic_params = ic_params or {}
        params = ic_params.get("f", {"type": "constant", "value": 0.0})
        # build x mesh and v mesh
        # use backend of spatial discretiser for x dimension
        backend = params.get("backend", getattr(self.diff, "_backend", "numpy"))
        # generate using helper for x coordinate only then broadcast to v
        from .base import _generate_field
        f_x = _generate_field(self.grid, params, backend=backend)
        # broadcast to velocity dimension
        if backend == "jax":
            import jax.numpy as jnp  # type: ignore[attr-defined]
            v_shape = (len(self.v_grid),)
            f = jnp.tile(f_x[..., None], (1,) * f_x.ndim + (len(self.v_grid),))
        else:
            import numpy as np
            f = np.tile(f_x[..., None], (1,) * f_x.ndim + (len(self.v_grid),))
        return {"f": f}

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        f = state["f"]
        # spatial derivative: use diff.grad along first axis
        # f has shape (nx, nv)
        # compute derivative along x only for each v
        df_dx = self.diff.grad(f)[0]
        # velocity derivative using central difference
        xp = self.diff._xp if hasattr(self.diff, "_xp") else _np
        # roll in v dimension (last axis)
        forward = xp.roll(f, -1, axis=-1)
        backward = xp.roll(f, 1, axis=-1)
        df_dv = (forward - backward) / (2.0 * self.dv)
        # boundaries: one‑sided
        # first v
        df_dv = df_dv.copy()
        df_dv[..., 0] = (f[..., 1] - f[..., 0]) / self.dv
        df_dv[..., -1] = (f[..., -1] - f[..., -2]) / self.dv
        rhs_f = - (self.v_grid[None, :] * df_dx + self.E * df_dv)
        return {"f": rhs_f}