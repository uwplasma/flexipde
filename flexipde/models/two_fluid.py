"""Two‑fluid plasma model (simplified).

This module provides a minimal two‑fluid model where ion and electron
density fields advect independently at prescribed velocities.  In a real
plasma the two fluids are coupled through charge neutrality, electric
fields and collisions; here we omit these effects for clarity and
demonstration of the framework.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .base import PDEModel


@dataclass
class TwoFluid(PDEModel):
    """A simplified two‑fluid model with separate advection velocities.

    Parameters
    ----------
    grid:
        The spatial grid (one or more dimensions).
    diff:
        The spatial discretiser.
    velocities:
        A sequence of two sequences; the first for ions and the second for
        electrons.  Each inner sequence has length equal to the number of
        spatial dimensions and specifies the constant advection velocity.
    """

    velocities: Sequence[Sequence[float]] = ((1.0,), (-1.0,))

    def __post_init__(self) -> None:
        self.field_names = ["n_i", "n_e"]
        super().__post_init__()

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        n_i = state["n_i"]
        n_e = state["n_e"]
        grads_i = self.diff.grad(n_i)
        grads_e = self.diff.grad(n_e)
        dudt_i = 0
        dudt_e = 0
        for v, g in zip(self.velocities[0], grads_i):
            dudt_i = dudt_i - v * g
        for v, g in zip(self.velocities[1], grads_e):
            dudt_e = dudt_e - v * g
        return {"n_i": dudt_i, "n_e": dudt_e}