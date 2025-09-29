r"""Example: Hasegawa–Wakatani system.

This script demonstrates how to implement and solve the Hasegawa–Wakatani
drift–wave system using flexipde.  The equations (simplified) are

.. math::
    \partial_t n + \partial_y \phi + \alpha (\phi - n) = 0,\\
    \partial_t \nabla^2 \phi + \partial_y n + \alpha (\phi - n) = 0.

The simulation is run on a periodic domain with Fourier discretisation.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.base import PDEModel
from flexipde.solver import Simulation


class HasegawaWakatani(PDEModel):
    def __init__(self, grid, diff, alpha: float):
        self.alpha = alpha
        super().__init__(grid=grid, diff=diff, linear=False)

    def __post_init__(self) -> None:
        # set field names and bcs then call base
        self.field_names = ["n", "phi"]
        self.field_bcs = ["periodic"] * self.grid.ndim
        super().__post_init__()

    def rhs(self, state, t):
        n = state["n"]
        phi = state["phi"]
        # gradients
        grad_n = self.diff.grad(n)
        grad_phi = self.diff.grad(phi)
        lap_phi = self.diff.laplacian(phi)
        # derivative along y axis is index 1 for 2D
        dn_dt = - grad_phi[1] - self.alpha * (phi - n)
        dphi_dt = - self.alpha * (phi - n) - grad_n[1]
        # The phi equation evolves lap_phi, so invert laplacian to get dphi/dt on phi
        return {"n": dn_dt, "phi": dphi_dt}


def main():
    # Domain and grid
    grid = Grid.regular([(0.0, 2.0 * np.pi), (0.0, 2.0 * np.pi)], [32, 32], [True, True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = HasegawaWakatani(grid, diff, alpha=0.5)
    # Initial density and potential: small random perturbation
    rng = np.random.default_rng(seed=0)
    ic = {
        "n": {"type": "sinusoidal", "amplitude": 0.1, "wavevector": [1, 1], "phase": 0.0},
        "phi": {"type": "sinusoidal", "amplitude": 0.1, "wavevector": [1, 1], "phase": np.pi / 2},
    }
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.02, save_every=10, initial_state_params=ic)
    result = sim.run()
    n0 = result.states[0]["n"]
    n_end = result.states[-1]["n"]
    x = grid.coords[0]
    y = grid.coords[1]
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(n0.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Initial density")
    im1 = axes[1].imshow(n_end.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Final density")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()