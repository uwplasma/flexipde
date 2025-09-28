"""
Base PDE models.

This module defines the :class:`PDEModel` base class which all models
should inherit from.  A model encapsulates a system of PDEs and
knows how to compute the time derivative of its state given the
current state and time.  Models should also handle boundary
conditions on their fields.

To define a new model, subclass :class:`PDEModel` and override the
methods :meth:`initial_state` and :meth:`rhs`.  The base class
provides utility functions for applying boundary conditions and
differentiation using the provided discretiser.

Notes
-----
* Fields are stored in a dictionary mapping field names to arrays.
* Boundary conditions are specified per field as a string ("periodic",
  "dirichlet" or "neumann") with optional constant value for
  Dirichlet conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Callable, Any, Sequence, Optional

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

from flexipde.grid import Grid
from flexipde.discretisation.base import BaseDifferentiator


@dataclass
class FieldBC:
    """Boundary condition specification for a field.

    Attributes
    ----------
    bc : str
        Type of boundary condition: ``"periodic"``, ``"dirichlet"`` or
        ``"neumann"``.
    value : float, optional
        Constant value for Dirichlet conditions.  Ignored for other
        types.
    """
    bc: str
    value: Optional[float] = None


@dataclass
class PDEModel:
    """Base class for PDE models.

    Parameters
    ----------
    grid : :class:`~flexipde.grid.Grid`
        The spatial grid.
    diff : :class:`~flexipde.discretisation.base.BaseDifferentiator`
        Discretisation scheme used to compute spatial derivatives.
    field_bcs : Dict[str, FieldBC], optional
        Boundary conditions for each field.  Defaults to periodic if not
        specified.
    """
    grid: Grid
    diff: BaseDifferentiator
    # Boundary conditions are not part of the constructor signature.  They
    # are set by models in ``__post_init__``.  Mark ``init=False`` to
    # avoid dataclass ordering conflicts when subclasses introduce
    # non‑default parameters after this default field.
    field_bcs: Dict[str, FieldBC] = field(default_factory=dict, init=False)

    def initial_state(self, ic_params: Optional[dict] = None) -> Dict[str, Any]:
        """Return initial state as a mapping from field names to arrays.

        Parameters
        ----------
        ic_params : dict, optional
            Optional dictionary specifying parameters for the initial
            condition.  Subclasses should interpret this mapping to
            customise the initial state (for example to set amplitude,
            phase or choose different functional forms).  When
            ``None`` (the default), models should provide a sensible
            standard initial condition.

        Returns
        -------
        dict
            Mapping of field names to initial arrays.

        Notes
        -----
        Subclasses should override this method.  The base class
        implementation simply raises :class:`NotImplementedError`.
        """
        raise NotImplementedError

    def rhs(self, state: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Compute time derivative of the state.

        Parameters
        ----------
        state : dict
            Current state mapping field names to arrays.
        t : float
            Current time.

        Returns
        -------
        dict
            Time derivative of each field.
        """
        raise NotImplementedError

    def apply_boundary_conditions(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply boundary conditions to a state.

        This method enforces Dirichlet or Neumann boundary conditions by
        modifying the state arrays in place.  Periodic boundaries are
        handled by the discretisation scheme and do not require
        modification.  For Dirichlet, ghost values are set to the
        specified constant.  For Neumann, the first derivative at the
        boundary is set to zero via copying the adjacent interior
        value.  If a field does not have an entry in ``field_bcs``, it
        defaults to periodic.
        """
        for name, arr in state.items():
            bc = self.field_bcs.get(name)
            if bc is None or bc.bc == "periodic":
                continue
            # Only apply BCs to numpy arrays; jax arrays use slice updates
            is_jax = _HAS_JAX and isinstance(arr, jax.Array)
            for axis in range(self.grid.dim):
                if bc.bc == "dirichlet":
                    val = bc.value if bc.value is not None else 0.0
                    # set first and last indices along axis to val
                    idx_first = [slice(None)] * arr.ndim
                    idx_first[axis] = 0
                    idx_last = [slice(None)] * arr.ndim
                    idx_last[axis] = -1
                    if is_jax:
                        arr = arr.at[tuple(idx_first)].set(val)
                        arr = arr.at[tuple(idx_last)].set(val)
                    else:
                        arr[tuple(idx_first)] = val
                        arr[tuple(idx_last)] = val
                elif bc.bc == "neumann":
                    # zero derivative: copy adjacent interior values
                    idx0 = [slice(None)] * arr.ndim
                    idx0[axis] = 0
                    idx1 = idx0.copy(); idx1[axis] = 1
                    idx_end = [slice(None)] * arr.ndim
                    idx_end[axis] = -1
                    idx_penult = idx_end.copy(); idx_penult[axis] = -2
                    if is_jax:
                        arr = arr.at[tuple(idx0)].set(arr[tuple(idx1)])
                        arr = arr.at[tuple(idx_end)].set(arr[tuple(idx_penult)])
                    else:
                        arr[tuple(idx0)] = arr[tuple(idx1)]
                        arr[tuple(idx_end)] = arr[tuple(idx_penult)]
            state[name] = arr
        return state

    @property
    def output_fields(self) -> Sequence[str]:
        """Return a list of field names in the state."""
        return list(self.field_bcs.keys()) if self.field_bcs else []

    # ------------------------------------------------------------------
    # Initial condition helper
    #
    # Several models such as ``LinearAdvection`` and ``Diffusion``
    # construct similar functional forms for their initial conditions.
    # To reduce duplication and make it easier for users to write new
    # models, this helper encapsulates the logic for generating a
    # scalar field on a regular grid according to common parameters.
    def _generate_initial_field(self, ic_params: Dict[str, Any], default_type: str = "gaussian") -> Any:
        """Generate a scalar field on this model's grid.

        Parameters
        ----------
        ic_params : dict
            Parameters controlling the functional form.  Recognised keys
            include ``type`` (``"gaussian"``, ``"sinusoidal"`` or
            ``"constant"``), ``amplitude``, ``wavevector``, ``phase`` and
            ``value``.  An optional ``backend`` key can force use of
            JAX or NumPy.  If omitted, the backend is chosen based on
            whether any supplied parameter is a JAX array.  The default
            type used when ``type`` is not provided is given by
            ``default_type``.
        default_type : str, optional
            Default initial condition type when none is provided.

        Returns
        -------
        array
            An array of shape ``self.grid.shape`` representing the
            initial scalar field.  The array is created using either
            ``numpy`` or ``jax.numpy`` depending on the determined backend.

        Notes
        -----
        This function does not handle multi‑field initial conditions or
        problems with non‑cartesian metrics (such as Vlasov models).  It
        should be used for single‑component scalar fields.  Models with
        multiple fields can call this function for each component as
        needed.
        """
        # Determine backend request
        backend = ic_params.get("backend") if ic_params else None
        use_jax = False
        if backend == "jax":
            use_jax = True
        elif backend == "numpy":
            use_jax = False
        else:
            # If any supplied parameter is a JAX array, use JAX
            if _HAS_JAX and ic_params:
                for key in ("amplitude", "phase", "value"):
                    val = ic_params.get(key)
                    if isinstance(val, jax.Array):
                        use_jax = True
                        break
                if not use_jax:
                    wv = ic_params.get("wavevector")
                    if isinstance(wv, (list, tuple)):
                        for k in wv:
                            if isinstance(k, jax.Array):
                                use_jax = True
                                break
        # Load coordinate arrays
        coords = self.grid.coordinate_arrays("jax" if (use_jax and _HAS_JAX) else "numpy")
        np_module = jnp if (use_jax and _HAS_JAX) else _np
        # Extract parameters with defaults
        ic_type = ic_params.get("type", default_type).lower() if ic_params else default_type.lower()
        amp: Any = ic_params.get("amplitude", 1.0) if ic_params else 1.0
        wavevector: Any = ic_params.get("wavevector", [1] * self.grid.dim) if ic_params else [1] * self.grid.dim
        phase: Any = ic_params.get("phase", 0.0) if ic_params else 0.0
        const_value: Any = ic_params.get("value", 0.0) if ic_params else 0.0
        # Promote wavevector to list
        if not isinstance(wavevector, (list, tuple)):
            wavevector = [wavevector] * self.grid.dim
        # Compute profile
        if ic_type == "gaussian":
            centre = [ (c.max() + c.min()) / 2.0 for c in coords ]
            sq = np_module.zeros(self.grid.shape)
            for c, c0 in zip(coords, centre):
                sq = sq + (c - c0) ** 2
            # Characteristic width: 10% of the largest domain extent
            sigma = 0.1 * max([float(coord.max() - coord.min()) for coord in coords])
            field = amp * np_module.exp(-sq / (2.0 * sigma**2))
        elif ic_type in {"sinusoidal", "sine"}:
            field = np_module.ones(self.grid.shape, dtype=amp.dtype if hasattr(amp, "dtype") else float)
            for c, k in zip(coords, wavevector):
                field = field * np_module.sin(k * c + phase)
            field = amp * field
        elif ic_type == "constant":
            field = np_module.full(self.grid.shape, const_value)
        else:
            raise ValueError(f"Unknown initial condition type '{ic_type}'")
        return field
