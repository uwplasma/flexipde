"""
grid
====

This module defines the :class:`Grid` class, which stores domain
information for PDE simulations.  A grid represents a structured set of
points in one or more dimensions and may carry metric information for
curvilinear coordinates.  The :class:`Grid` does not perform any
differencing itself; instead it is passed to discretisation routines
defined in :mod:`flexipde.discretisation`.

Users can construct a :class:`Grid` directly by specifying coordinate
arrays and a metric tensor, or by loading from a configuration file
using :func:`Grid.from_config`.  The metric is stored as an array with
shape ``(dim, dim) + shape``, where ``dim`` is the number of spatial
dimensions and ``shape`` is the shape of the grid.  A diagonal metric
with all ones corresponds to Cartesian coordinates.

Reference
---------
* BOUT++ emphasises the separation of geometry from the equations,
  allowing arbitrary curvilinear coordinates and runtime‑swappable
  numerical methods【501252340464299†L24-L33】.  This grid class plays a similar
  role in flexipde.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Sequence, Any
import numpy as _np

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    # Fallback for environments without jax; some functionality may be
    # unavailable.  We still provide a minimal stub for type hints.
    jax = None  # type: ignore
    jnp = None  # type: ignore


@dataclass
class Grid:
    """A structured grid with optional metric information.

    Parameters
    ----------
    coordinates : Tuple[_np.ndarray, ...] or Tuple[jnp.ndarray, ...]
        A tuple of coordinate arrays, one for each spatial dimension.  Each
        array should have shape ``shape``.  For example, in a 2D
        Cartesian grid with dimensions ``(nx, ny)`` you might supply
        ``(x, y)`` where ``x`` has shape ``(nx, ny)`` and stores the x
        coordinate at each grid point.
    metric : Optional[_np.ndarray] or jnp.ndarray
        A metric tensor with shape ``(dim, dim) + shape``.  If
        ``None``, the metric is taken to be the identity (Cartesian
        coordinates).  For a diagonal metric you can supply an array
        with shape ``(dim,) + shape`` which will be promoted to a full
        tensor internally.
    periodic : Sequence[bool], optional
        Flags indicating whether each dimension has periodic boundary
        conditions by default.  Boundary conditions can be overridden
        per field in the model configuration.
    """

    coordinates: Tuple[Any, ...]
    metric: Optional[Any] = None
    periodic: Tuple[bool, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.dim = len(self.coordinates)
        if not self.periodic:
            self.periodic = tuple(False for _ in range(self.dim))
        # Expand diagonal metric if necessary
        if self.metric is None:
            # Identity metric
            self.metric = None
        else:
            if isinstance(self.metric, (list, tuple)):
                self.metric = _np.array(self.metric)
            # Promote diagonal metric to full tensor
            if self.metric.ndim == self.dim + 1:
                # metric has shape (dim,) + shape
                diag = self.metric
                full = _np.zeros((self.dim, self.dim) + diag.shape[1:], dtype=diag.dtype)
                for i in range(self.dim):
                    full[(i, i)] = diag[i]
                self.metric = full

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the number of points along each dimension."""
        return self.coordinates[0].shape

    @property
    def ndim(self) -> int:
        return self.dim

    def coordinate_arrays(self, backend: str = "numpy") -> Tuple[Any, ...]:
        """Return the coordinate arrays, optionally converting to JAX.

        Parameters
        ----------
        backend : str, optional
            Either ``"numpy"`` or ``"jax"``.  If ``"jax"``, the
            coordinates are converted to JAX arrays via ``jnp.asarray``.
        """
        if backend == "jax":
            if jax is None:
                raise RuntimeError("JAX is not available; cannot convert coordinates to jax arrays")
            return tuple(jnp.asarray(c) for c in self.coordinates)
        return self.coordinates

    @classmethod
    def regular(cls, domain: Sequence[Tuple[float, float]], shape: Sequence[int], *,
                periodic: Optional[Sequence[bool]] = None) -> "Grid":
        """Construct a regular Cartesian grid.

        Parameters
        ----------
        domain : Sequence of (float, float)
            Lower and upper bounds for each dimension.
        shape : Sequence of int
            Number of points along each dimension.
        periodic : Sequence of bool, optional
            Whether each dimension is periodic.  Defaults to ``False`` in all
            directions.
        """
        coords = []
        for (lo, hi), n in zip(domain, shape):
            x = _np.linspace(lo, hi, n, endpoint=False)
            # broadcast to full shape
            arr = x
            # reshape to broadcast along other axes
            reshape_dims = [1] * len(shape)
            reshape_dims[len(coords)] = n
            arr = arr.reshape(reshape_dims)
            coords.append(_np.broadcast_to(arr, shape))
        return cls(tuple(coords), metric=None, periodic=tuple(periodic) if periodic else tuple(False for _ in shape))

    @classmethod
    def from_config(cls, cfg: dict) -> "Grid":
        """Construct a grid from a configuration dictionary.

        The configuration should include at least ``domain`` and ``shape``.
        Optionally, ``metric`` and ``periodic`` can be specified.

        Example TOML snippet::

            [grid]
            domain = [[0.0, 1.0], [0.0, 1.0]]  # 2D box from (0,0) to (1,1)
            shape = [128, 128]
            periodic = [true, true]

        """
        domain = cfg.get("domain")
        shape = cfg.get("shape")
        periodic = cfg.get("periodic")
        metric = cfg.get("metric")
        if domain is None or shape is None:
            raise ValueError("Grid config must provide 'domain' and 'shape'")
        return cls.regular(domain, shape, periodic=periodic)

    def spacing(self) -> Tuple[Any, ...]:
        """Compute grid spacing along each dimension.

        This method returns a tuple of spacing values ``dx_i`` for each
        dimension by computing the difference between the first two
        points along that axis.  It avoids slicing issues for higher
        dimensions by using ``numpy.take`` directly.

        Returns
        -------
        tuple
            Spacing ``dx_i`` for each dimension.
        """
        dxs: list[Any] = []
        for axis, arr in enumerate(self.coordinates):
            # Each coordinate array has the full grid shape.  Take the
            # first two points along the given axis to compute the
            # spacing.  Using ``take`` avoids reducing the dimensionality
            # unexpectedly for higher dimensions.
            # Ensure there are at least two points
            if arr.shape[axis] < 2:
                raise ValueError("Need at least two points along each dimension to compute spacing")
            first = _np.take(arr, 0, axis=axis)
            second = _np.take(arr, 1, axis=axis)
            dxs.append(second - first)
        return tuple(dxs)

    def to_jax(self) -> "Grid":
        """Return a new Grid with JAX arrays for coordinates and metric."""
        if jax is None:
            raise RuntimeError("JAX is not available; cannot convert grid to jax")
        coords = tuple(jnp.asarray(c) for c in self.coordinates)
        metric = None if self.metric is None else jnp.asarray(self.metric)
        return Grid(coords, metric=metric, periodic=self.periodic)

    def to_config(self) -> dict:
        """Return a configuration dictionary representing this grid.

        The returned dictionary contains keys ``domain``, ``shape`` and
        ``periodic`` so that the grid can be reconstructed via
        :meth:`Grid.from_config`.  The domain bounds are inferred from
        the minimum and maximum coordinate values along each axis.  If
        the grid is periodic, the upper bound is estimated as the last
        grid point plus the spacing to the next (which is assumed
        uniform).
        """
        domain = []
        shape = list(self.shape)
        periodic = list(self.periodic)
        for axis, arr in enumerate(self.coordinates):
            lo = float(arr.min())
            hi = float(arr.max())
            # approximate upper bound for periodic grids
            if periodic[axis]:
                # estimate spacing along axis
                # take first two points
                slicer = [0] * self.dim
                slicer[axis] = slice(0, 2)
                axis_vals = arr[tuple(slicer)]
                if axis_vals.shape[axis] >= 2:
                    dx = float(_np.take(axis_vals, 1, axis=axis) - _np.take(axis_vals, 0, axis=axis))
                    hi += dx
            domain.append([lo, hi])
        return {"domain": domain, "shape": shape, "periodic": periodic}
