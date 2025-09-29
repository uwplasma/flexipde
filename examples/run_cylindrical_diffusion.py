"""Example: diffusion in cylindrical coordinates.

This script demonstrates a diffusion problem on a 2D domain
representing $(r,\theta)$ coordinates.  The radial direction uses
Dirichlet boundaries while the azimuthal direction is periodic.  A
Gaussian spot diffuses over time.  Note that flexipde does not yet
include a full cylindrical Laplacian, so this example approximates the
operator using the finite difference differentiator.
"""
import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models import Diffusion
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 1.0), (0.0, 2 * np.pi)], [64, 64], [False, True])
    diff = FiniteDifference(grid)
    model = Diffusion(grid, diff, diffusivity=0.05)
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.001)
    sim.initial_state_params = {
        "u": {"type": "gaussian", "amplitude": 1.0, "center": [0.5, np.pi], "width": [0.1, 0.3]}
    }
    result = sim.run()
    u0 = result.states[0]["u"]
    u1 = result.states[-1]["u"]
    r = grid.coords[0]
    theta = grid.coords[1]
    R, T = np.meshgrid(r, theta, indexing='ij')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axs[0].imshow(u0, extent=(0, 1, 0, 2 * np.pi), origin='lower', aspect='auto')
    axs[0].set_title("Initial u")
    fig.colorbar(im0, ax=axs[0])
    im1 = axs[1].imshow(u1, extent=(0, 1, 0, 2 * np.pi), origin='lower', aspect='auto')
    axs[1].set_title("Final u")
    fig.colorbar(im1, ax=axs[1])
    plt.suptitle("Diffusion in cylindrical coordinates (approx)")
    plt.show()


if __name__ == "__main__":
    main()