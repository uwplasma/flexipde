"""Example script for linear advection.

This script demonstrates how to set up and run a one‑dimensional
linear advection problem using the high‑level Python API.  It
constructs a periodic grid, a spectral discretiser on NumPy, an
advection model with unit velocity, and then runs the simulation
over one advection period.  After completion it prints a simple
summary of the final state.  Users can modify the grid size,
velocity or initial conditions to experiment with different
scenarios.

Run this script with::

    python examples/run_advection.py

"""

from __future__ import annotations

import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.advection import LinearAdvection
from flexipde.solver import Simulation


def main() -> None:
    # Create a 1D periodic grid on [0, 2π) with 128 points
    grid = Grid.regular([(0.0, 2 * np.pi)], [128], periodic=[True])
    # Use a spectral discretiser on NumPy backend
    diff = SpectralDifferentiator(grid, backend="numpy")
    # Advection model with velocity v=1.0
    model = LinearAdvection(grid, diff, velocity=[1.0])
    # Set up simulation from t=0 to t=2 (one full period)
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.01, save_every=10)
    # Run the simulation
    times, states = sim.run()
    # Report the range of u at final time
    final_u = states[-1]["u"]
    print(f"Completed advection simulation: t = {times[-1]:.2f}, min u = {final_u.min():.3f}, max u = {final_u.max():.3f}")


if __name__ == "__main__":
    main()
