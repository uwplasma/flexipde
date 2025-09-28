"""Example script for an ideal MHD Alfvén wave.

This example sets up a simple one–dimensional Alfvén wave using the
``IdealAlfven`` model.  The background magnetic field ``B0`` is
constant and the initial perturbation is sinusoidal.  The wave
propagates along the domain at the Alfvén speed.  After running
the simulation, the script prints the maximum of the magnetic and
velocity perturbations at the final time.

Run this script with::

    python examples/run_alfven.py

"""

from __future__ import annotations

import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.ideal_mhd import IdealAlfven
from flexipde.solver import Simulation


def main() -> None:
    """Run a one‑dimensional Alfvén wave simulation and report results."""
    # 1D periodic domain [0, 2π)
    grid = Grid.regular([(0.0, 2 * np.pi)], [128], periodic=[True])
    # Spectral differentiation for MHD on NumPy backend
    diff = SpectralDifferentiator(grid, backend="numpy")
    # Background magnetic field B0 = [1.0] along x
    model = IdealAlfven(grid, diff, B0=[1.0])
    # simulation over one period (the Alfven speed here is 1)
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.01, save_every=10)
    times, states = sim.run()
    final_state = states[-1]
    # Extract the transverse velocity and magnetic field components.  The
    # IdealAlfven model stores components as ``v0``, ``v1`` (and
    # ``B0``, ``B1``) depending on dimensionality.  In one dimension the
    # relevant fields are ``v0`` and ``B0``.  We index the first (and
    # only) component for convenience.
    v0 = final_state["v0"]
    B0 = final_state["B0"]
    print(
        f"Completed Alfven simulation: t = {times[-1]:.2f}, max |v| = {np.abs(v0).max():.3f}, max |B| = {np.abs(B0).max():.3f}",
    )


if __name__ == "__main__":
    main()
