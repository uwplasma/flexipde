"""Grid utilities.

The :class:`~flexipde.grid.Grid` class represents a structured multi‑dimensional
grid.  It stores coordinate arrays for each spatial dimension and provides
helpers to compute uniform spacing.  Only structured Cartesian or
curvilinear (via metric tensors) grids are currently supported.  The grid
information is immutable once constructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as _np


@dataclass
class Grid:
    """A structured grid.

    Parameters
    ----------
    coords:
        A list of coordinate arrays, one per spatial dimension.  Each array
        should have shape ``(n,)`` where ``n`` is the number of points in that
        dimension.  The arrays are stored by reference and should not be
        modified after construction.
    periodic:
        A list of booleans indicating whether each dimension uses periodic
        boundary conditions.  Periodic dimensions will automatically apply
        periodic wrapping for finite difference stencils.
    metric:
        An optional metric tensor for curvilinear coordinates.  This is a
        callable returning a matrix ``g_{ij}`` at each point.  It is not
        currently used directly by the discretisation classes but is provided
        for future extensions.
    """

    coords: List[Any]
    periodic: Sequence[bool]
    metric: Any | None = None

    @property
    def ndim(self) -> int:
        """Return the number of spatial dimensions."""
        return len(self.coords)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the grid (number of points per dimension)."""
        return tuple(len(c) for c in self.coords)

    @property
    def coords_arrays(self) -> List[Any]:
        """Alias for the list of coordinate arrays."""
        return self.coords

    def spacing(self) -> Tuple[float, ...]:
        """Return the uniform spacing for each dimension.

        If coordinate arrays are uniformly spaced, compute the spacing from
        successive differences.  For periodic dimensions the spacing is
        computed by dividing the domain length by the number of points.
        """
        spacings: List[float] = []
        for c, periodic in zip(self.coords, self.periodic):
            if len(c) < 2:
                spacings.append(1.0)
                continue
            if periodic:
                # For periodic grids assume that c spans one full period from
                # c[0] to c[-1] + spacing.  Compute spacing as total length / N.
                total_len = float(c[-1] - c[0] + (c[1] - c[0]))
                spacings.append(total_len / len(c))
            else:
                # Non‑periodic uniform spacing from first two points
                spacings.append(float(c[1] - c[0]))
        return tuple(spacings)

    @staticmethod
    def regular(domain: Sequence[Tuple[float, float]],
                shape: Sequence[int],
                periodic: Sequence[bool]) -> 'Grid':
        """Construct a uniformly spaced grid on the given domain.

        Parameters
        ----------
        domain:
            A sequence of ``(lower, upper)`` tuples giving the extent of each
            dimension.  For periodic dimensions, the upper bound is inclusive
            and the spacing is determined by dividing the total domain length
            by the number of points.
        shape:
            The number of grid points in each dimension.
        periodic:
            Whether each dimension is periodic.

        Returns
        -------
        Grid
        """
        if len(domain) != len(shape) or len(shape) != len(periodic):
            raise ValueError("domain, shape and periodic must all have the same length")
        coords: List[Any] = []
        for (a, b), n, per in zip(domain, shape, periodic):
            if n < 1:
                raise ValueError("Number of grid points must be >= 1")
            if per:
                # Include the end point implicitly by leaving out the last point
                x = _np.linspace(a, b, n, endpoint=False)
            else:
                x = _np.linspace(a, b, n, endpoint=True)
            coords.append(x)
        return Grid(coords=coords, periodic=list(periodic))

    def to_config(self) -> dict:
        """Serialize the grid to a dictionary for saving.

        Only the domain extents and periodic flags are stored.  This is used
        when saving simulation results so that the grid can be reconstructed.
        """
        domain: List[Tuple[float, float]] = []
        spacings = self.spacing()
        for c, per, dx in zip(self.coords, self.periodic, spacings):
            if per:
                # Domain length equals N * dx
                domain.append((float(c[0]), float(c[0] + dx * len(c))))
            else:
                domain.append((float(c[0]), float(c[-1])))
        return {
            'domain': domain,
            'shape': list(self.shape),
            'periodic': list(self.periodic),
        }