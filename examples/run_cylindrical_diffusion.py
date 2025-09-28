"""Axisymmetric diffusion in cylindrical coordinates.

This example demonstrates how to define a custom PDE model using
flexipde to solve the diffusion equation in cylindrical coordinates
$(r,z)$ assuming axisymmetry (no $\theta$ dependence).  The
Laplacian in this case is

.. math::

    \nabla^2 u = \frac{1}{r} \frac{\partial}{\partial r}\left(r \frac{\partial u}{\partial r}\right) + \frac{\partial^2 u}{\partial z^2},

which can be expanded as $\partial^2 u/\partial r^2 + (1/r) \partial u/\partial r + \partial^2 u/\partial z^2$.  We discretise
the $r$ and $z$ directions on a regular grid and compute the
derivatives using finite differences.  A Gaussian peak at the
origin is allowed to diffuse over time.

Run this script with::

    python examples/run_cylindrical_diffusion.py

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Sequence, Dict

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.base import PDEModel, FieldBC
from flexipde.solver import Simulation


class CylindricalDiffusion(PDEModel):
    """Custom diffusion model in cylindrical $(r,z)$ coordinates."""

    diffusivity: float = 1.0
    init_u: Any | None = None  # Optional custom initialiser

    def __post_init__(self) -> None:
        # Default boundary conditions: zero value at boundaries
        self.field_bcs = {"u": FieldBC("dirichlet", value=0.0)}

    def initial_state(self, ic_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # Use custom function if provided
        if self.init_u is not None:
            coords = self.grid.coordinate_arrays("numpy")
            u0 = self.init_u(coords)
            return {"u": u0}
        # Otherwise use Gaussian centred at (r=0.0,z=0.5)
        coords = self.grid.coordinate_arrays("numpy")
        r, z = coords
        sigma_r = 0.1 * (r.max() - r.min())
        sigma_z = 0.1 * (z.max() - z.min())
        u0 = np.exp(-((r) ** 2 / (2.0 * sigma_r ** 2) + (z - 0.5) ** 2 / (2.0 * sigma_z ** 2)))
        return {"u": u0}

    def rhs(self, state: Dict[str, Any], t: float) -> Dict[str, Any]:
        # Extract field
        u = state["u"]
        # Obtain coordinate arrays and discretiser
        coords = self.grid.coordinate_arrays("numpy")
        r = coords[0]
        # First derivative in r
        du_dr = self.diff.grad(u, 0)
        # Second derivative in r
        d2u_dr2 = self.diff.grad(du_dr, 0)
        # Second derivative in z
        d2u_dz2 = self.diff.grad(self.diff.grad(u, 1), 1)
        # Avoid division by zero at r=0 by adding a small epsilon
        # Extract a column of r for broadcasting along z (shape (nr,1)).
        # The grid radial coordinate starts at 0.01 so division is safe.
        r_safe = r[:, :1]
        radial_term = d2u_dr2 + du_dr / r_safe
        lap = radial_term + d2u_dz2
        return {"u": self.diffusivity * lap}


def main() -> None:
    # Cylindrical domain: r ∈ [0,1], z ∈ [0,1]; 64×64 grid points
    # Non‑periodic in r, periodic in z
    # Shift the radial coordinate away from zero to avoid singular behaviour at r=0
    grid = Grid.regular([(0.01, 1.0), (0.0, 1.0)], [64, 64], periodic=[False, True])
    diff = FiniteDifference(grid, backend="numpy")
    model = CylindricalDiffusion(grid, diff)
    # Set the diffusion coefficient manually (the dataclass default is 1.0)
    model.diffusivity = 0.05
    # For cylindrical problems it's common to use Neumann BC at r=0 (symmetry)
    # but here we keep the default Dirichlet BCs.  Users can adjust via:
    #   model.field_bcs["u"] = FieldBC("neumann")
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.0001, save_every=20)
    times, states = sim.run()
    u0 = states[0]["u"]
    uf = states[-1]["u"]
    print(
        f"Completed cylindrical diffusion: t = {times[-1]:.3f}, min u = {uf.min():.4f}, max u = {uf.max():.4f}",
    )
    # Plot the radial profile at z=0.5 for the initial and final state
    z_axis = grid.coordinates[1]
    # find index closest to z=0.5
    idx_z = int(len(z_axis) / 2)
    r_coords = grid.coordinates[0]
    plt.figure()
    plt.plot(r_coords, u0[:, idx_z], label="initial (z=0.5)")
    plt.plot(r_coords, uf[:, idx_z], label="final (z=0.5)")
    plt.xlabel("r")
    plt.ylabel("u(r, z=0.5)")
    plt.title("Cylindrical Diffusion (axisymmetric)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
