"""Base classes for PDE models.

All models inherit from :class:`PDEModel`.  A model must implement
``rhs(state, t)`` that returns the time derivative of the state as a
dictionary of arrays.  Optionally it can override ``initial_state`` to
generate the initial condition from parameter dictionaries.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..grid import Grid
from ..discretisation import SpectralDifferentiator, FiniteDifference


def _generate_field(grid: Grid, ic_params: Dict[str, Any], backend: str = "numpy") -> Any:
    """Utility to generate a field from initial condition parameters.

    Supported keys in ``ic_params``:

    * ``type``: one of ``"constant"``, ``"gaussian"``, ``"sinusoidal"``.  If omitted,
      ``constant`` is used.
    * For constant: ``value``.
    * For gaussian: ``amplitude``, ``center`` (list of floats), ``width`` (scalar or list), and optional ``backend``.
    * For sinusoidal: ``amplitude``, ``wavevector`` (list of ints), ``phase``.

    The returned array has shape ``grid.shape`` and will be a NumPy or JAX array depending on ``backend``.
    """
    # choose xp
    if backend == "jax":
        try:
            import jax.numpy as jnp  # type: ignore[attr-defined]
        except Exception:
            import numpy as jnp  # fallback to numpy if JAX missing
    else:
        import numpy as jnp  # type: ignore[assignment]
    # default type; allow passing a raw array via key 'array'
    if "array" in ic_params:
        # user provided array directly; convert to appropriate backend array
        arr = ic_params["array"]
        if backend == "jax":
            try:
                import jax.numpy as jnp  # type: ignore[attr-defined]
            except Exception:
                import numpy as jnp  # type: ignore[assignment]
        else:
            import numpy as jnp  # type: ignore[assignment]
        return jnp.array(arr)
    kind = ic_params.get("type", "constant")
    if kind == "constant":
        val = ic_params.get("value", 0.0)
        field = jnp.zeros(grid.shape)
        field = field + val
        return field
    elif kind == "gaussian":
        amp = ic_params.get("amplitude", 1.0)
        center = ic_params.get("center", [0.0] * grid.ndim)
        width = ic_params.get("width", 1.0)
        # if width is a single value or string expression, replicate for each dim
        if not isinstance(width, (list, tuple)):
            width = [width] * grid.ndim
        # convert center and width values to floats, evaluating simple expressions if strings
        center_vals = []
        width_vals = []
        for c in center:
            if isinstance(c, str):
                try:
                    # import here to avoid circular import
                    from ..io import _safe_eval  # type: ignore
                    c_val = _safe_eval(c)
                except Exception:
                    # fallback: try to cast directly
                    c_val = float(c)
            else:
                c_val = float(c)
            center_vals.append(c_val)
        for w in width:
            if isinstance(w, str):
                try:
                    from ..io import _safe_eval  # type: ignore
                    w_val = _safe_eval(w)
                except Exception:
                    w_val = float(w)
            else:
                w_val = float(w)
            width_vals.append(w_val)
        # build meshgrid and compute Gaussian
        coords = grid.coords
        mesh = jnp.meshgrid(*coords, indexing='ij')
        expr = 1.0
        for xi, c0, w in zip(mesh, center_vals, width_vals):
            expr = expr * jnp.exp(-((xi - c0) ** 2) / (2.0 * (w ** 2)))
        return amp * expr
    elif kind == "sinusoidal":
        amp = ic_params.get("amplitude", 1.0)
        kvec = ic_params.get("wavevector", [1] * grid.ndim)
        phase = ic_params.get("phase", 0.0)
        coords = grid.coords
        mesh = jnp.meshgrid(*coords, indexing='ij')
        arg = 0.0
        for xi, k in zip(mesh, kvec):
            arg = arg + k * xi
        return amp * jnp.sin(arg + phase)
    else:
        raise ValueError(f"Unknown initial condition type: {kind}")


@dataclass
class PDEModel:
    """Base class for PDE models.

    Subclasses must override :meth:`rhs` to compute time derivatives.  They
    may also override :meth:`initial_state` to define custom initial
    conditions.
    """

    grid: Grid
    diff: Any  # discretisation object
    # names of fields stored in state dict; subclasses should set this in __post_init__
    field_names: Sequence[str] = field(default_factory=list, init=False)
    # boundary condition type for each dimension and field
    field_bcs: List[str] = field(default_factory=list, init=False)
    # mark linear PDEs; used for manufactured tests
    linear: bool = True

    def __post_init__(self) -> None:
        if not self.field_bcs:
            # default all periodic
            self.field_bcs = ["periodic"] * self.grid.ndim

    def apply_bcs(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply boundary conditions to all fields.

        For periodic boundaries this is a no‑op.  Dirichlet boundaries
        currently reflect values at the edges; Neumann boundaries copy
        adjacent values.  This method returns a new dict but does not
        modify the input arrays.
        """
        # For finite difference schemes only; spectral is inherently periodic.
        if isinstance(self.diff, SpectralDifferentiator):
            return state
        new_state: Dict[str, Any] = {}
        for field, arr in state.items():
            a = arr
            # apply along each axis if needed
            for axis, bc in enumerate(self.field_bcs):
                if bc == "periodic":
                    continue
                if bc == "dirichlet":
                    # set boundary values to zero
                    # do a copy to avoid modifying original
                    a = a.copy()
                    sl0 = [slice(None)] * a.ndim
                    sl0[axis] = 0
                    a[tuple(sl0)] = 0.0
                    sl0[axis] = -1
                    a[tuple(sl0)] = 0.0
                elif bc == "neumann":
                    # copy interior values to boundary
                    a = a.copy()
                    sl0 = [slice(None)] * a.ndim
                    sl0[axis] = 0
                    sl1 = sl0.copy()
                    sl1[axis] = 1
                    a[tuple(sl0)] = a[tuple(sl1)]
                    sl0[axis] = -1
                    sl1[axis] = -2
                    a[tuple(sl0)] = a[tuple(sl1)]
            new_state[field] = a
        return new_state

    def initial_state(self, ic_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Create the initial state for the simulation.

        Parameters
        ----------
        ic_params:
            A mapping from field name to a dictionary of initial condition
            parameters.  If ``None`` or a field is missing, a constant zero
            field is used.

        Returns
        -------
        dict
            A dictionary mapping field names to arrays with shape
            ``grid.shape``.
        """
        state: Dict[str, Any] = {}
        ic_params = ic_params or {}
        for name in self.field_names:
            params = ic_params.get(name, {"type": "constant", "value": 0.0})
            backend = params.get("backend", getattr(self.diff, "_backend", "numpy"))
            state[name] = _generate_field(self.grid, params, backend=backend)
        return state

    def rhs(self, state: Dict[str, Any], t: Any) -> Dict[str, Any]:  # pragma: no cover
        """Compute the time derivative of the state at time ``t``.

        Subclasses must override this method.
        """
        raise NotImplementedError