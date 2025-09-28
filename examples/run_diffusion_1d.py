"""One‑dimensional diffusion with Neumann boundary conditions.

This script demonstrates how to solve a 1D diffusion equation on a
non‑periodic domain with Neumann (zero‑gradient) boundaries.  It
constructs a regular grid, uses a finite‑difference discretiser and
specifies a Gaussian initial condition.  At the end of the simulation
it prints basic statistics of the final state and plots the initial
and final profiles for quick visual inspection.

Run this script with::

    python examples/run_diffusion_1d.py

"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.diffusion import Diffusion
from flexipde.models.base import FieldBC
from flexipde.solver import Simulation


def main() -> None:
    # Domain [0, 1] with 128 grid points and non‑periodic boundaries
    grid = Grid.regular([(0.0, 1.0)], [128], periodic=[False])
    # Finite‑difference discretiser on NumPy backend
    diff = FiniteDifference(grid, backend="numpy")
    # Gaussian initial condition centred at 0.5 with width 0.1
    def init_gauss(coords: list[Any]) -> np.ndarray:
        x = coords[0]
        return np.exp(-((x - 0.5) ** 2) / (2 * (0.1**2)))
    model = Diffusion(grid, diff, diffusivity=0.1, init_u=init_gauss)
    # Set Neumann (zero gradient) boundary conditions on the scalar field
    model.field_bcs["u"] = FieldBC("neumann")
    # Set up and run the simulation
    # Use a small time step for stability of the explicit Euler solver
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.0001, save_every=20)
    times, states = sim.run()
    u0 = states[0]["u"]
    uf = states[-1]["u"]
    # Print summary
    print(
        f"Completed 1D diffusion: t = {times[-1]:.3f}, min u = {uf.min():.4f}, max u = {uf.max():.4f}",
    )
    # Plot the initial and final profiles for a quick look
    plt.figure()
    plt.plot(grid.coordinates[0], u0, label="initial")
    plt.plot(grid.coordinates[0], uf, label="final")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("1D Diffusion with Neumann BCs")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
