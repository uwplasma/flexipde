"""Example: Resistive MHD.

This script runs the simplified resistive magnetohydrodynamics model and
plots the transverse velocity and magnetic field at the initial and final
times.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from flexipde.io import build_simulation


def main() -> None:
    import pathlib
    cfg_path = pathlib.Path(__file__).resolve().parent / "resistive_mhd.toml"
    sim = build_simulation(str(cfg_path))
    result = sim.run()
    times = result.times
    states = result.states
    x = sim.model.grid.coords[0]
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, states[0]["v"], label=f"t={times[0]:.2f}")
    plt.plot(x, states[-1]["v"], label=f"t={times[-1]:.2f}")
    plt.title("v(x,t)")
    plt.xlabel("x")
    plt.ylabel("v")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, states[0]["B"], label=f"t={times[0]:.2f}")
    plt.plot(x, states[-1]["B"], label=f"t={times[-1]:.2f}")
    plt.title("B(x,t)")
    plt.xlabel("x")
    plt.ylabel("B")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()