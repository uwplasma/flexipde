"""
Ideal MHD model.

This module implements a simplified form of ideal magnetohydrodynamics (MHD)
for demonstration purposes.  The model evolves the transverse components of
velocity and magnetic field under a uniform background magnetic field
``B0``.  It solves

.. math::

    \partial_t v_i &= \sum_j B_{0j} \partial_j B_i,\\
    \partial_t B_i &= \sum_j B_{0j} \partial_j v_i,

where ``v_i`` and ``B_i`` are the components of the velocity and
perturbation magnetic field respectively, and ``B_{0j}`` are the
components of the constant background field.  Density and pressure
effects are ignored.  In one dimension with ``B0=(1,)`` this reduces
to the classic Alfvén wave equation used to verify MHD codes.

This model is not intended to be a full replacement for a general
MHD code but serves as a template for how to assemble a coupled
system of equations using flexipde.  Extending to full MHD would
require adding continuity, momentum and induction equations as well
as pressure and energy evolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Callable

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
class IdealAlfven(PDEModel):
    """Ideal Alfvén wave model.

    Parameters
    ----------
    grid : :class:`~flexipde.grid.Grid`
        The spatial grid.  All dimensions should be periodic for the
        spectral discretisation, but finite differences may be used for
        non‑periodic dimensions.
    diff : discretiser
        Differentiator used to compute spatial derivatives.
    B0 : Sequence[float]
        Components of the constant background magnetic field.  Length
        must match ``grid.dim``.  Defaults to (1.0,) in one dimension.
    init_v : Callable, optional
        Function mapping coordinate arrays to initial velocity
        components.  Should accept a tuple of arrays (one per
        dimension) and return a sequence of arrays of length ``dim``.
        If not provided, defaults to a transverse sinusoidal wave in
        the first component.
    init_B : Callable, optional
        Function mapping coordinate arrays to initial magnetic field
        perturbation.  Similar semantics to ``init_v``.
    """

    B0: Sequence[float] = (1.0,)
    init_v: Callable[[Sequence[Any]], Sequence[Any]] | None = None
    init_B: Callable[[Sequence[Any]], Sequence[Any]] | None = None

    def __post_init__(self) -> None:
        # Set default boundary conditions: periodic for v and B components
        bcs: Dict[str, FieldBC] = {}
        for i in range(self.grid.dim):
            bcs[f"v{i}"] = FieldBC("periodic")
            bcs[f"B{i}"] = FieldBC("periodic")
        self.field_bcs = bcs

    def initial_state(self, ic_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Generate the initial state for the ideal Alfvén model.

        This supports optional parameters ``ic_params`` with the
        following keys:

        * ``amplitude``: amplitude of the initial sinusoidal wave. May
          be a Python float or a JAX array.
        * ``phase``: phase shift (in radians) of the sinusoid.
        * ``type``: currently only ``"sinusoidal"`` is supported; other
          values are ignored.
        * ``backend``: Optional string ``"jax"`` or ``"numpy"`` to
          force the use of a particular backend when constructing
          initial conditions.  If not provided, the backend is
          inferred from the type of amplitude or the availability of
          JAX.

        Custom callables ``init_v`` and ``init_B`` passed at
        construction override these parameters.  When provided, the
        user functions should accept coordinate arrays using either
        NumPy or JAX and return arrays of the same backend.

        Returns
        -------
        dict
            Mapping from field names to their initial arrays.
        """
        # Determine amplitude and phase
        amp: Any = 1.0
        phase: Any = 0.0
        backend: Optional[str] = None
        if ic_params:
            amp = ic_params.get("amplitude", amp)
            phase = ic_params.get("phase", phase)
            backend = ic_params.get("backend")
        # Choose backend heuristically
        use_jax = False
        if backend == "jax":
            use_jax = True
        elif backend == "numpy":
            use_jax = False
        else:
            if _HAS_JAX:
                # If amplitude or phase is a JAX array use JAX
                if isinstance(amp, jax.Array) or isinstance(phase, jax.Array):
                    use_jax = True
        # Load coordinates for chosen backend
        coords = self.grid.coordinate_arrays("jax" if (use_jax and _HAS_JAX) else "numpy")
        np_module = jnp if (use_jax and _HAS_JAX) else _np
        dim = self.grid.dim
        # Velocity components
        if self.init_v is None:
            v = [ np_module.zeros(self.grid.shape, dtype=float) for _ in range(dim) ]
            # Choose index 0 for 1D, or 1 for higher dimensions
            idx = 0 if dim == 1 else 1
            v[idx] = amp * np_module.sin(coords[0] + phase)
        else:
            # User‑provided function; call with appropriate coords
            v = list(self.init_v(coords))
        # Magnetic field components
        if self.init_B is None:
            B = [ np_module.zeros(self.grid.shape, dtype=float) for _ in range(dim) ]
            idx = 0 if dim == 1 else 1
            B[idx] = amp * np_module.sin(coords[0] + phase)
        else:
            B = list(self.init_B(coords))
        # Assemble state dictionary
        state: Dict[str, Any] = {}
        for i in range(dim):
            state[f"v{i}"] = v[i]
            state[f"B{i}"] = B[i]
        return state

    def rhs(self, state: Dict[str, Any], t: float) -> Dict[str, Any]:
        # Compute time derivative for each component
        dim = self.grid.dim
        dstate: Dict[str, Any] = {}
        # Extract arrays into lists for convenience
        v = [state[f"v{i}"] for i in range(dim)]
        B = [state[f"B{i}"] for i in range(dim)]
        # compute derivatives
        for i in range(dim):
            # dv_i/dt = sum_j B0[j] * ∂B_i/∂x_j
            dv = None
            dB = None
            for j, B0j in enumerate(self.B0):
                if B0j == 0:
                    continue
                # derivative along axis j of B[i]
                dB_i = self.diff.grad(B[i], j)
                dV_i = self.diff.grad(v[i], j)
                dv = dB_i * B0j if dv is None else dv + dB_i * B0j
                dB = dV_i * B0j if dB is None else dB + dV_i * B0j
            dstate[f"v{i}"] = dv
            dstate[f"B{i}"] = dB
        return dstate
