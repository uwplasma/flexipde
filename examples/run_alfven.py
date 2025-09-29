"""Example: 1D ideal Alfvén wave simulation.

We simulate a transverse Alfvén wave propagating along a uniform
background magnetic field.  The model is linear and involves two fields,
the transverse velocity and magnetic field perturbation.  The wave travels
undistorted at the Alfvén speed.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import IdealAlfven
from flexipde.solver import Simulation


def main() -> None:
    L = 2.0 * np.pi
    grid = Grid.regular([(0.0, L)], [128], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = IdealAlfven(grid, diff, B0=1.0)
    # Initial perturbations: v = -sin(x), B = sin(x)
    ic = {
        "v": {"type": "sinusoidal", "amplitude": -1.0, "wavevector": [1], "phase": 0.0},
        "B": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0},
    }
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.1, save_every=10, initial_state_params=ic)
    result = sim.run()
    x = grid.coords[0]
    v0 = result.states[0]["v"]
    v_end = result.states[-1]["v"]
    plt.figure()
    plt.plot(x, v0, label="Initial v")
    plt.plot(x, v_end, label="Final v")
    plt.xlabel("x")
    plt.ylabel("v_y")
    plt.legend()
    plt.title("Ideal Alfvén wave")
    plt.show()


if __name__ == "__main__":
    main()