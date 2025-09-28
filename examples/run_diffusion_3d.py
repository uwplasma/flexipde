"""Three‑dimensional diffusion on a periodic cube.

This script solves the diffusion equation on a 3D periodic domain.  A
Gaussian blob is initialised at the centre of the cube and allowed to
spread out.  Periodic boundary conditions are applied in all
directions.  The script prints summary statistics and visualises a
2D cross‑section of the initial and final states.

Run this script with::

    python examples/run_diffusion_3d.py

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.diffusion import Diffusion
from flexipde.solver import Simulation


def main() -> None:
    # 3D periodic domain [0,1]^3 with 32^3 grid points
    grid = Grid.regular([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], [32, 32, 32], periodic=[True, True, True])
    diff = FiniteDifference(grid, backend="numpy")
    # Gaussian initial condition at centre
    def init_gauss(coords: list[np.ndarray]) -> np.ndarray:
        x, y, z = coords
        sigma = 0.1
        return np.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2) / (2.0 * sigma ** 2)))
    model = Diffusion(grid, diff, diffusivity=0.01, init_u=init_gauss)
    # Run simulation for a short time
    # Use a smaller time step for stability in 3D
    sim = Simulation(model, t0=0.0, t1=0.05, dt0=0.0001, save_every=10)
    times, states = sim.run()
    u0 = states[0]["u"]
    uf = states[-1]["u"]
    # Print summary statistics
    print(
        f"Completed 3D diffusion: t = {times[-1]:.3f}, min u = {uf.min():.4f}, max u = {uf.max():.4f}",
    )
    # Visualise a 2D slice through the centre in the z direction
    mid = u0.shape[2] // 2
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    im0 = ax0.imshow(u0[:, :, mid], extent=(0, 1, 0, 1), origin="lower")
    ax0.set_title("Initial (z midplane)")
    fig.colorbar(im0, ax=ax0)
    im1 = ax1.imshow(uf[:, :, mid], extent=(0, 1, 0, 1), origin="lower")
    ax1.set_title("Final (z midplane)")
    fig.colorbar(im1, ax=ax1)
    fig.suptitle("3D Diffusion (cross‑section)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
