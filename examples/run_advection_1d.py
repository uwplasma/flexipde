"""Example: 1D linear advection.

This driver script constructs a 1D grid, spectral differentiator and linear
advection model, sets a sinusoidal initial condition and integrates the
equation in time.  At the end it plots the initial and final state on
screen.
"""
import pathlib

import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import LinearAdvection
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 2 * np.pi)], [128], [True])
    diff = SpectralDifferentiator(grid)
    model = LinearAdvection(grid, diff, velocity=[1.0])
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.01)
    sim.initial_state_params = {"u": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0}}
    result = sim.run()
    x = grid.coords[0]
    u0 = result.states[0]["u"]
    u1 = result.states[-1]["u"]
    plt.figure()
    plt.plot(x, u0, label="initial")
    plt.plot(x, u1, label="final")
    plt.xlabel("x")
    plt.ylabel("u")
    plt.legend()
    plt.title("1D Linear Advection")
    plt.show()


if __name__ == "__main__":
    main()