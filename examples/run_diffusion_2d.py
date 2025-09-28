"""Two‑dimensional diffusion with Dirichlet boundary conditions.

This script solves the diffusion equation on a unit square using
non‑periodic boundaries.  Dirichlet boundary conditions (fixed zero
value) are applied on all edges.  A Gaussian hot spot is initialised
at the centre and allowed to diffuse.  The script prints a summary
of the final state and displays a colour map of the initial and final
temperature fields.

Run this script with::

    python examples/run_diffusion_2d.py

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Sequence

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.diffusion import Diffusion
from flexipde.models.base import FieldBC
from flexipde.solver import Simulation


def main() -> None:
    # 2D domain [0,1]×[0,1] with 64×64 points and non‑periodic boundaries
    grid = Grid.regular([(0.0, 1.0), (0.0, 1.0)], [64, 64], periodic=[False, False])
    diff = FiniteDifference(grid, backend="numpy")
    # Gaussian initial condition centred at (0.5, 0.5)
    def init_gauss(coords: Sequence[Any]) -> np.ndarray:
        x, y = coords
        sigma = 0.1
        return np.exp(-(((x - 0.5) ** 2 + (y - 0.5) ** 2) / (2.0 * sigma ** 2)))
    model = Diffusion(grid, diff, diffusivity=0.05, init_u=init_gauss)
    # Apply Dirichlet boundary conditions (u=0 at edges)
    model.field_bcs["u"] = FieldBC("dirichlet", value=0.0)
    # Run simulation for a short time
    # Use a smaller time step for stability of the explicit Euler solver
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.0001, save_every=20)
    times, states = sim.run()
    u0 = states[0]["u"]
    uf = states[-1]["u"]
    print(
        f"Completed 2D diffusion: t = {times[-1]:.3f}, min u = {uf.min():.4f}, max u = {uf.max():.4f}",
    )
    # Plot initial and final states
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))
    im0 = ax0.imshow(u0, extent=(0, 1, 0, 1), origin="lower")
    ax0.set_title("Initial")
    fig.colorbar(im0, ax=ax0)
    im1 = ax1.imshow(uf, extent=(0, 1, 0, 1), origin="lower")
    ax1.set_title("Final")
    fig.colorbar(im1, ax=ax1)
    fig.suptitle("2D Diffusion with Dirichlet BCs")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
