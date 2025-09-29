"""Example: 2D diffusion on a square domain.

This script runs a two–dimensional diffusion equation with Dirichlet
boundaries on a unit square.  The finite–difference discretisation is used.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models import Diffusion
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 1.0), (0.0, 1.0)], [32, 32], [False, False])
    diff = FiniteDifference(grid, backend="numpy")
    model = Diffusion(grid, diff, diffusivity=0.1)
    ic = {"u": {"type": "constant", "value": 1.0}}
    sim = Simulation(model, t0=0.0, t1=0.5, dt0=0.01, save_every=10, initial_state_params=ic)
    result = sim.run()
    u0 = result.states[0]["u"]
    u_end = result.states[-1]["u"]
    x = grid.coords[0]
    y = grid.coords[1]
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(u0.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Initial u")
    im1 = axes[1].imshow(u_end.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Final u")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()