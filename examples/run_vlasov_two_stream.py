"""Example: two‑stream instability.

This driver sets up the 1D Vlasov–Poisson two‑stream instability
simulation using the built‑in Vlasov model.  Because the Vlasov
equation is high‑dimensional, the script does not plot the full phase
space distribution but instead shows the density (velocity integral)
and electric field at the final time.
"""
import numpy as np
import matplotlib.pyplot as plt

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import VlasovTwoStream
from flexipde.solver import Simulation


def main() -> None:
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid)
    model = VlasovTwoStream(
        grid,
        diff,
        nv=64,
        v_min=-5.0,
        v_max=5.0,
        amplitude=0.05,
        drift_velocity=2.0,
        thermal_velocity=0.5,
        background_density=0.5,
    )
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.1)
    # no additional initial state parameters; the model sets up its own initial distribution
    result = sim.run()
    f_final = result.states[-1]["f"]  # shape (nx, nv)
    # compute density by integrating over velocity axis
    rho = np.trapz(f_final, axis=1)
    x = grid.coords[0]
    plt.figure()
    plt.plot(x, rho)
    plt.xlabel("x")
    plt.ylabel("density")
    plt.title("Two‑stream instability final density")
    plt.show()


if __name__ == "__main__":
    main()