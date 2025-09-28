"""
Diffusion equation.

This module implements a simple diffusion model for a scalar field
``u`` governed by

.. math::

    \partial_t u = D \nabla^2 u,

where ``D`` is a constant diffusivity.  The diffusion equation is
parabolic and stiff when the spatial resolution is high; thus an
implicit time integrator may be desirable.  Users can select an
implicit solver (e.g. Crank–Nicolson or backward Euler) through the
Diffrax interface when instantiating the solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

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
class Diffusion(PDEModel):
    """Linear diffusion of a scalar field."""
    diffusivity: float = 1.0
    init_u: callable | None = None

    def __post_init__(self) -> None:
        self.field_bcs = {"u": FieldBC("periodic")}

    def initial_state(self, ic_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate the initial state for the diffusion equation.

        This implementation mirrors that of :class:`LinearAdvection` but
        uses a sine wave as the default functional form when no
        parameters are provided.  It leverages the
        :meth:`PDEModel._generate_initial_field` helper to avoid code
        duplication.  The same ``ic_params`` keys are supported.

        Parameters
        ----------
        ic_params : dict, optional
            Dictionary of initial condition parameters.  If omitted or
            empty, a sine wave along the first axis is used.

        Returns
        -------
        dict
            Mapping from ``"u"`` to the initial array.
        """
        # Custom user initialiser
        if self.init_u is not None and (ic_params is None or not isinstance(ic_params, dict)):
            coords = self.grid.coordinate_arrays("numpy")
            u0 = self.init_u(coords)
            return {"u": u0}
        params = ic_params if ic_params is not None else {}
        # If no parameters are provided, default to sine wave along first axis
        if not params:
            # Determine backend: use JAX if available so the result is differentiable
            use_jax = _HAS_JAX
            coords = self.grid.coordinate_arrays("jax" if use_jax else "numpy")
            np_module = jnp if (use_jax and _HAS_JAX) else _np
            u0 = np_module.sin(coords[0])
            return {"u": u0}
        # Otherwise generate using helper.  Use a Gaussian by default if
        # type is not specified in params; this matches previous default.
        field = self._generate_initial_field(params, default_type="gaussian")
        return {"u": field}

    def rhs(self, state: Dict[str, Any], t: float) -> Dict[str, Any]:
        u = state["u"]
        lap = self.diff.laplacian(u)
        return {"u": self.diffusivity * lap}
