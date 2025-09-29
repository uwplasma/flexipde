r"""Example: custom 2D advection on a sphere.

This example demonstrates how to define a custom PDE model outside the
``flexipde`` package.  We solve a simple advection equation on the
surface of a sphere in terms of the coordinates ``(θ, φ)``.  The
equation is

.. math::

    \partial_t u + v_\theta \frac{\partial u}{\partial \theta}
        + \frac{v_\phi}{\sin\theta} \frac{\partial u}{\partial \phi} = 0.

We use a finite difference discretisation on a ``θ``--``φ`` grid.  The
velocities ``v_θ`` and ``v_φ`` are constants in this simple example.

To run this example, execute:

.. code-block:: bash

    python run_spherical_advection_custom.py

Note that this custom model is defined entirely within this script;
users can create their own models similarly without modifying
``flexipde`` itself.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.base import PDEModel
from flexipde.solver import Simulation


class SphericalAdvection(PDEModel):
    """Custom 2D advection equation on a sphere.

    Parameters
    ----------
    grid:
        The ``θ, φ`` grid.  The first dimension corresponds to polar angle
        ``θ \in [0, \pi]`` and the second to azimuthal angle ``φ \in [0, 2\pi)``.
    diff:
        A ``FiniteDifference`` discretiser on this grid.
    velocities:
        A sequence of two floats ``(v_θ, v_φ)`` giving the constant
        advection velocities in the ``θ`` and ``φ`` directions.
    """

    def __init__(self, grid: Grid, diff: FiniteDifference, velocities: tuple[float, float]) -> None:
        self.velocities = velocities
        super().__init__(grid, diff)

    def __post_init__(self) -> None:
        self.field_names = ["u"]
        super().__post_init__()

    def rhs(self, state: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:
        u = state["u"]
        # gradients: list of arrays [du/dθ, du/dφ]
        grads = self.diff.grad(u)
        dtheta = grads[0]
        dphi = grads[1]
        v_theta, v_phi = self.velocities
        # sin(theta) for the metric factor; broadcast to grid shape
        theta = self.grid.coords[0][:, None]  # shape (nθ, 1)
        sin_theta = np.sin(theta)
        # avoid division by zero at the poles by clipping
        sin_theta = np.where(sin_theta == 0.0, 1.0, sin_theta)
        du_dt = -v_theta * dtheta - v_phi * (dphi / sin_theta)
        return {"u": du_dt}


def main() -> None:
    # Define a spherical grid: θ in [0, π], φ in [0, 2π)
    n_theta = 64
    n_phi = 128
    grid = Grid.regular([(0.0, np.pi), (0.0, 2.0 * np.pi)], [n_theta, n_phi], [False, True])
    diff = FiniteDifference(grid)
    model = SphericalAdvection(grid, diff, velocities=(0.5, 1.0))
    # Initial condition: Gaussian blob centred at (θ0, φ0)
    theta0 = np.pi / 3.0
    phi0 = np.pi
    width = 0.2
    theta = grid.coords[0][:, None]
    phi = grid.coords[1][None, :]
    u0 = np.exp(-((theta - theta0) ** 2 + (phi - phi0) ** 2) / (2 * width ** 2))
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.01)
    # supply initial condition via initial_state_params
    # Provide initial condition as a raw array via the "array" key so that
    # flexipde copies it directly.
    sim.initial_state_params = {"u": {"array": u0}}
    result = sim.run()
    # Plot initial and final states
    u_initial = result.states[0]["u"]
    u_final = result.states[-1]["u"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(u_initial, origin="lower", aspect="auto")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Initial u")
    im1 = axes[1].imshow(u_final, origin="lower", aspect="auto")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Final u at t=1")
    plt.suptitle("Custom spherical advection")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()