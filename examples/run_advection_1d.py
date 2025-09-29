"""Example: 1D advection equation simulation.

This script runs a one‑dimensional linear advection equation on a periodic
domain using a spectral discretiser.  The initial condition is a
sinusoidal wave.  The result is plotted after a few periods to show
propagation without distortion.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import Advection
from flexipde.solver import Simulation


def main() -> None:
    # Periodic domain of length 2π
    L = 2.0 * np.pi
    grid = Grid.regular([(0.0, L)], [128], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = Advection(grid, diff, velocity=[1.0])
    # Sinusoidal initial condition: u(x,0) = sin(x)
    ic = {"u": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0}}
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.1, save_every=10, initial_state_params=ic)
    result = sim.run()
    x = grid.coords[0]
    u0 = result.states[0]["u"]
    uf = result.states[-1]["u"]
    plt.figure()
    plt.plot(x, u0, label="Initial u")
    plt.plot(x, uf, label="Final u")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u")
    plt.title("1D Advection")
    plt.show()


if __name__ == "__main__":
    main()