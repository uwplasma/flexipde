r"""Diffusion equation.

The diffusion model evolves a scalar field ``u`` according to

.. math::

    \partial_t u = D \nabla^2 u,

where ``D`` is the diffusivity.  The Laplacian is computed by the chosen
discretiser.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import PDEModel


@dataclass
class Diffusion(PDEModel):
    """A linear diffusion model for a scalar field.

    Parameters
    ----------
    grid:
        The spatial grid.
    diff:
        The spatial discretiser.
    diffusivity:
        The diffusion coefficient ``D``.  Defaults to ``1.0``.
    """

    diffusivity: float = 1.0

    def __post_init__(self) -> None:
        self.field_names = ["u"]
        super().__post_init__()

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        u = state["u"]
        lap = self.diff.laplacian(u)
        return {"u": self.diffusivity * lap}