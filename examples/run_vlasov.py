"""Example script for the two–stream instability (Vlasov–Poisson).

This example sets up a one–dimensional Vlasov–Poisson two–stream
instability using the ``VlasovTwoStream`` model.  Two drifting
Maxwellian beams are initialised with a small sinusoidal density
perturbation.  The script evolves the system for a short time and
prints the electric field amplitude at the final time.

Run this script with::

    python examples/run_vlasov.py

"""

from __future__ import annotations

import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.vlasov import VlasovTwoStream
from flexipde.solver import Simulation


def main() -> None:
    # 1D periodic physical domain and velocity domain parameters
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], periodic=[True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    # Create Vlasov model with 32 velocity grid points in [-5,5]
    model = VlasovTwoStream(grid, diff, nv=32, v_min=-5.0, v_max=5.0)
    # Simulation parameters
    sim = Simulation(model, t0=0.0, t1=0.5, dt0=0.005, save_every=10)
    # Set initial conditions: amplitude 0.1, mode 1
    sim.initial_state_params_list = [
        {
            "amplitude": 0.1,
            "mode": 1,
            "drift_velocity": 1.0,
            "thermal_velocity": 1.0,
            "background_density": 1.0,
        }
    ]
    (times, states) = sim.run()
    final_f = states[-1]["f"]
    # Compute electric field from distribution
    # Private method _poisson_field returns E(x) given f(x,v)
    E = model._poisson_field(final_f, np)
    print(
        f"Completed Vlasov simulation: t = {times[-1]:.2f}, max |E| = {np.abs(E).max():.3f}",
    )


if __name__ == "__main__":
    main()
