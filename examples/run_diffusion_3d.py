"""Example: 3D diffusion on a periodic cube.

This script runs a three–dimensional diffusion equation with periodic
boundaries.  Because visualising a full 3D field is challenging, we
compute and print the mean value of the field at the initial and final
time.
"""
from __future__ import annotations

import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models import Diffusion
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], [16, 16, 16], [True, True, True])
    diff = FiniteDifference(grid, backend="numpy")
    model = Diffusion(grid, diff, diffusivity=0.05)
    ic = {"u": {"type": "constant", "value": 1.0}}
    sim = Simulation(model, t0=0.0, t1=0.2, dt0=0.01, save_every=5, initial_state_params=ic)
    result = sim.run()
    u0 = result.states[0]["u"]
    u_end = result.states[-1]["u"]
    print("Initial mean:", np.mean(u0))
    print("Final mean:", np.mean(u_end))


if __name__ == "__main__":
    main()