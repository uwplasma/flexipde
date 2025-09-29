r"""Resistive MHD example model.

This class implements a very simple 1D resistive magnetohydrodynamics (MHD)
system.  It is **not** a complete MHD solver but serves as a demonstration
of how to include additional physics such as resistivity.  The equations
are a reduced form of the induction and momentum equations:

.. math::

    \partial_t v_y &= \frac{d B_y}{d x},\\
    \partial_t B_y &= \frac{d v_y}{d x} + \eta \nabla^2 B_y,

where ``η`` is the resistivity.  A constant background field and density
are implicitly set to one.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import PDEModel


@dataclass
class ResistiveMHD(PDEModel):
    """A simple resistive MHD model with two scalar fields ``v`` and ``B``.

    Parameters
    ----------
    grid:
        The spatial grid (1D).
    diff:
        The spatial discretiser.
    eta:
        Resistivity parameter; defaults to 0.01.
    """

    eta: float = 0.01

    def __post_init__(self) -> None:
        self.field_names = ["v", "B"]
        super().__post_init__()

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        v = state["v"]
        B = state["B"]
        grads_v = self.diff.grad(v)
        grads_B = self.diff.grad(B)
        dv_dt = grads_B[0]  # derivative of B
        dB_dt = grads_v[0] + self.eta * self.diff.laplacian(B)
        return {"v": dv_dt, "B": dB_dt}