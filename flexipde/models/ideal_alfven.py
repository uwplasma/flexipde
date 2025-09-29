r"""Ideal Alfvén wave model.

This is a toy model for the propagation of shear Alfvén waves in one
dimension.  It evolves transverse velocity ``v`` and magnetic field ``B``
according to

.. math::

    \partial_t v_y &= \partial_x B_y,\\
    \partial_t B_y &= \partial_x v_y.

The solution describes Alfvén waves travelling at unit speed.  The
variables ``v`` and ``B`` remain equal (up to a sign) at all times.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import PDEModel


@dataclass
class IdealAlfven(PDEModel):
    """Ideal Alfvén wave model.

    Parameters
    ----------
    grid:
        The spatial grid (1D).
    diff:
        The spatial discretiser.
    """

    # Optional constant background field B0 for extensibility.
    B0: float = 1.0

    def __post_init__(self) -> None:
        self.field_names = ["v", "B"]
        super().__post_init__()

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        v = state["v"]
        B = state["B"]
        grads_v = self.diff.grad(v)
        grads_B = self.diff.grad(B)
        dv_dt = grads_B[0]
        dB_dt = grads_v[0]
        return {"v": dv_dt, "B": dB_dt}