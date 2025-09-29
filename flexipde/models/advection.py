r"""Linear advection equation.

This module implements a simple linear advection model where a scalar field
``u`` is transported at a constant velocity.  The equation in conservative
form is

.. math::

    \partial_t u + \sum_i v_i \partial_{x_i} u = 0.

The solution is simply the initial profile translated by ``v_i t`` in each
direction.  Periodic boundaries are assumed by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .base import PDEModel


@dataclass
class LinearAdvection(PDEModel):
    """A linear advection model for a single scalar field ``u``.

    Parameters
    ----------
    grid:
        The spatial grid.
    diff:
        The spatial discretiser (spectral or finite difference).
    velocity:
        A sequence of floats giving the constant advection velocity in each
        dimension.  The length must match the dimensionality of the grid.
    """

    velocity: Sequence[float] = (1.0,)

    def __post_init__(self) -> None:
        self.field_names = ["u"]
        super().__post_init__()

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        u = state["u"]
        grads = self.diff.grad(u)
        dudt = 0
        for v, g in zip(self.velocity, grads):
            dudt = dudt - v * g
        return {"u": dudt}