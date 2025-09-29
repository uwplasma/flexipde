"""Example: Vlasov two–stream instability simulation.

This script runs a one–dimensional Vlasov–Poisson system with two drifting
beams (two–stream instability).  The initial distribution consists of a
superposition of two shifted Maxwellians plus a small sinusoidal perturbation.
We compute the time evolution and plot the spatial density at the final
time.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import VlasovTwoStream
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 2.0 * np.pi)], [32], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = VlasovTwoStream(grid, diff, nv=64, v_min=-5.0, v_max=5.0)
    ic = {
        "f": {
            "amplitude": 0.05,
            "drift_velocity": 1.0,
            "thermal_velocity": 1.0,
            "background_density": 1.0,
        }
    }
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.05, save_every=5, initial_state_params=ic)
    result = sim.run()
    # compute spatial density rho(x) = \int f dv at final time
    f_end = result.states[-1]["f"]
    dv = (model.v_max - model.v_min) / model.nv
    rho = np.sum(f_end, axis=1) * dv
    x = grid.coords[0]
    plt.plot(x, rho)
    plt.xlabel("x")
    plt.ylabel("Density ρ")
    plt.title("Two–stream instability: final density")
    plt.show()


if __name__ == "__main__":
    main()