"""
Linear advection equation.

This module implements a simple advection model for a scalar field ``u``
carried by a constant velocity field ``v``.  The governing equation is

.. math::

    \partial_t u + \mathbf{v}\cdot\nabla u = 0,

where ``v`` is a constant vector.  This is the classic test
equation used to verify numerical schemes; its analytic solution is
just a translation of the initial condition at speed ``v``.

The model supports arbitrary dimensions; ``v`` must have length equal
to the number of spatial dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

from .base import PDEModel, FieldBC


@dataclass
class LinearAdvection(PDEModel):
    """Linear advection of a scalar field by a constant velocity."""

    velocity: Sequence[float]
    init_u: callable | None = None

    def __post_init__(self) -> None:
        # Set default BC: periodic for u
        self.field_bcs = {"u": FieldBC("periodic")}
        # Ensure velocity length matches grid dimension
        if len(self.velocity) != self.grid.dim:
            raise ValueError(f"Velocity dimension {len(self.velocity)} does not match grid.dim {self.grid.dim}")

    def initial_state(self, ic_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate the initial state for the advection equation.

        This method delegates the construction of the scalar field to
        :meth:`PDEModel._generate_initial_field`.  It supports the same
        ``ic_params`` keys as before (``type``, ``amplitude``,
        ``wavevector``, ``phase``, ``value`` and optional ``backend``).
        If no parameters are supplied, a Gaussian bump is used by
        default.  A user‑provided callable ``init_u`` overrides the
        built‑in generator.

        Parameters
        ----------
        ic_params : dict, optional
            Dictionary controlling the initial condition.  If ``None``,
            defaults are used.

        Returns
        -------
        dict
            Mapping from ``"u"`` to the initial array.
        """
        # If a custom initializer is provided and no parameter dict
        if self.init_u is not None and (ic_params is None or not isinstance(ic_params, dict)):
            coords = self.grid.coordinate_arrays("jax" if (_HAS_JAX and isinstance(self.velocity, Sequence) and any(isinstance(v, jax.Array) for v in self.velocity)) else "numpy")
            u0 = self.init_u(coords)
            return {"u": u0}
        # Otherwise use the helper function
        params = ic_params if ic_params is not None else {}
        field = self._generate_initial_field(params, default_type="gaussian")
        return {"u": field}

    def rhs(self, state: Dict[str, Any], t: float) -> Dict[str, Any]:
        u = state["u"]
        du_dt = None
        for axis, vj in enumerate(self.velocity):
            if vj == 0:
                continue
            grad_u = self.diff.grad(u, axis)
            term = -vj * grad_u
            du_dt = term if du_dt is None else du_dt + term
        return {"u": du_dt}
