"""Example: 1D diffusion equation.

This script loads the ``diffusion_1d.toml`` configuration, runs the
simulation and plots the initial and final profiles.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from flexipde.io import build_simulation


def main() -> None:
    # Resolve the TOML file relative to this script
    import pathlib
    cfg_path = pathlib.Path(__file__).resolve().parent / "diffusion_1d.toml"
    sim = build_simulation(str(cfg_path))
    result = sim.run()
    times = result.times
    states = result.states
    x = sim.model.grid.coords[0]
    plt.figure()
    plt.plot(x, states[0]["u"], label=f"t={times[0]:.2f}")
    plt.plot(x, states[-1]["u"], label=f"t={times[-1]:.2f}")
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.legend()
    plt.title("1D Diffusion")
    plt.show()


if __name__ == '__main__':
    main()