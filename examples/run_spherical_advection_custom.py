"""Example: 2D advection on a sphere.

This custom example demonstrates how to implement a new coordinate
system outside of the built‑in grids.  We solve a 2D advection
equation on the surface of the unit sphere using spherical
coordinates ``(θ, φ)``.  The PDE is

.. math::

    \partial_t u + v_\theta \frac{\partial u}{\partial \theta}
        + v_\phi \frac{1}{\sin\theta} \frac{\partial u}{\partial \phi} = 0,

where ``v_θ`` and ``v_φ`` are constant velocities.  The grid is
periodic in both directions for simplicity.  This script defines a
custom :class:`PDEModel` subclass with its own spatial derivatives and
integrates it with the :class:`Simulation` wrapper.

To run the example:

.. code-block:: bash

    python examples/run_spherical_advection_custom.py

The script will plot the initial and final distributions on the
sphere.  No TOML file is used because the user defines the model
directly in Python.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.base import PDEModel
from flexipde.solver import Simulation


class SphericalAdvection(PDEModel):
    """Custom 2D advection on a sphere.

    Parameters
    ----------
    grid:
        A 2D grid with domains ``[(0, π), (0, 2π)]`` corresponding to
        ``θ`` and ``φ``.  Periodic boundaries should be used on both
        axes.
    diff:
        A dummy finite difference discretiser.  It is not used in the
        RHS calculation but satisfies the base class signature.
    velocity:
        Tuple ``(v_theta, v_phi)`` giving the constant angular
        velocities.
    """

    def __init__(self, grid: Grid, diff: Any, velocity: tuple[float, float] = (1.0, 0.5)) -> None:
        super().__init__(grid, diff)
        self.velocity = velocity

    def __post_init__(self) -> None:
        # One field named 'u'; periodic boundary conditions in both directions
        self.field_names = ["u"]
        self.field_bcs = ["periodic", "periodic"]
        super().__post_init__()

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        u = state["u"]
        v_theta, v_phi = self.velocity
        # grid spacings in θ and φ
        dtheta, dphi = self.grid.spacing()
        # central differences with periodic boundaries
        du_dtheta = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dtheta)
        du_dphi = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dphi)
        theta = self.grid.coords[0][:, None]
        # Avoid division by zero at the poles by adding a small epsilon
        eps = 1e-6
        sin_theta = np.sin(theta) + eps
        # Advection terms
        adv_theta = v_theta * du_dtheta
        adv_phi = v_phi * (du_dphi / sin_theta)
        return {"u": -(adv_theta + adv_phi)}


def main() -> None:
    # Create a 2D grid in θ ∈ [0, π] and φ ∈ [0, 2π]
    grid = Grid.regular([(0.0, np.pi), (0.0, 2.0 * np.pi)], [64, 128], [True, True])
    diff = FiniteDifference(grid)  # unused but required by constructor
    model = SphericalAdvection(grid, diff, velocity=(1.0, 0.25))
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.01)
    # initial condition: a bump near the equator
    theta, phi = np.meshgrid(grid.coords[0], grid.coords[1], indexing="ij")
    u0 = np.exp(-((theta - np.pi / 2)**2) / 0.2**2) * (1 + 0.5 * np.cos(phi))
    sim.initial_state_params = {"u": {"array": u0}}
    result = sim.run()
    # Extract results
    u_initial = result.states[0]["u"]
    u_final = result.states[-1]["u"]
    # Plot initial and final on a 2D contour plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].pcolormesh(grid.coords[1], grid.coords[0], u_initial, shading="auto")
    axes[0].set_title("Initial u(θ, φ)")
    axes[0].set_xlabel("φ")
    axes[0].set_ylabel("θ")
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].pcolormesh(grid.coords[1], grid.coords[0], u_final, shading="auto")
    axes[1].set_title("Final u(θ, φ)")
    axes[1].set_xlabel("φ")
    axes[1].set_ylabel("θ")
    fig.colorbar(im1, ax=axes[1])
    fig.suptitle("Spherical advection with constant angular velocities")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()