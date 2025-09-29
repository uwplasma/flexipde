"""Example: 1D ideal Alfvén waves.

This script simulates the propagation of shear Alfvén waves in one
dimension using the spectral differentiator.  The initial perturbation
consists of equal and opposite sine waves in the velocity and magnetic
fields.  The solution should propagate without distortion.
"""
import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import IdealAlfven
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 2 * np.pi)], [128], [True])
    diff = SpectralDifferentiator(grid)
    model = IdealAlfven(grid, diff)
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.01)
    sim.initial_state_params = {
        "v": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0},
        "B": {"type": "sinusoidal", "amplitude": -1.0, "wavevector": [1], "phase": 0.0},
    }
    result = sim.run()
    x = grid.coords[0]
    v0 = result.states[0]["v"]
    v1 = result.states[-1]["v"]
    plt.figure()
    plt.plot(x, v0, label="v initial")
    plt.plot(x, v1, label="v final")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.legend()
    plt.title("1D Ideal Alfvén Wave")
    plt.show()


if __name__ == "__main__":
    main()