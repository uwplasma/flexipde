"""Example: 2D diffusion equation.

This script solves the diffusion equation on a 2D square with
non‑periodic (Dirichlet) boundaries using the finite difference
discretisation.  The initial condition is a Gaussian bump at the
centre of the domain.  The script plots the initial and final
temperature fields.
"""
import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models import Diffusion
from flexipde.solver import Simulation


def main() -> None:
    # 2D domain [0,1]x[0,1]
    grid = Grid.regular([(0.0, 1.0), (0.0, 1.0)], [64, 64], [False, False])
    diff = FiniteDifference(grid)
    model = Diffusion(grid, diff, diffusivity=0.1)
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.001)
    sim.initial_state_params = {
        "u": {
            "type": "gaussian",
            "amplitude": 1.0,
            "center": [0.5, 0.5],
            "width": 0.1,
        }
    }
    result = sim.run()
    u0 = result.states[0]["u"]
    u1 = result.states[-1]["u"]
    x = grid.coords[0]
    y = grid.coords[1]
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(u0, extent=(0, 1, 0, 1), origin='lower')
    axs[0].set_title("Initial u")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(u1, extent=(0, 1, 0, 1), origin='lower')
    axs[1].set_title("Final u")
    fig.colorbar(im1, ax=axs[1])
    plt.suptitle("2D Diffusion")
    plt.show()


if __name__ == "__main__":
    main()