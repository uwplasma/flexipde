"""Example: 3D diffusion equation.

This script solves the diffusion equation in three dimensions on a unit
cube using finite differences.  A Gaussian hot spot at the centre
spreads over time.  Because visualising 3D data is challenging, the
script prints the maximum and minimum values of the field at the final
time and displays a 2D slice through the centre.
"""
import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models import Diffusion
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)], [32, 32, 32], [False, False, False])
    diff = FiniteDifference(grid)
    model = Diffusion(grid, diff, diffusivity=0.1)
    sim = Simulation(model, t0=0.0, t1=0.05, dt0=0.001)
    sim.initial_state_params = {
        "u": {"type": "gaussian", "amplitude": 1.0, "center": [0.5, 0.5, 0.5], "width": 0.1}
    }
    result = sim.run()
    u0 = result.states[0]["u"]
    u1 = result.states[-1]["u"]
    # print summary
    print("Initial max/min:", u0.max(), u0.min())
    print("Final max/min:", u1.max(), u1.min())
    # slice through middle along z
    mid = u1.shape[2] // 2
    slice2d = u1[:, :, mid]
    x = grid.coords[0]
    y = grid.coords[1]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(slice2d, extent=(0, 1, 0, 1), origin='lower')
    ax.set_title("u at z=0.5, t=0.05")
    fig.colorbar(im, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()