r"""Example: custom 2D Burgers' equation.

This script demonstrates how to define a custom PDE model outside of
flexipde's built‑in models.  We solve the 2D viscous Burgers' equation

.. math::
    \partial_t u + u \partial_x u + v \partial_y u = \nu \nabla^2 u,

on a periodic domain, where ``v`` is a constant velocity in the y
direction.  The model inherits from :class:`flexipde.models.base.PDEModel`.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.base import PDEModel
from flexipde.solver import Simulation


class Burgers2D(PDEModel):
    def __init__(self, grid, diff, nu: float, vy: float):
        self.nu = nu
        self.vy = vy
        # call base initializer with only grid and diff; field_names and field_bcs will be set in __post_init__
        super().__init__(grid=grid, diff=diff)

    def __post_init__(self) -> None:
        # assign field names and boundary conditions then call base
        self.field_names = ["u"]
        self.field_bcs = ["periodic"] * self.grid.ndim
        super().__post_init__()

    def rhs(self, state, t):
        u = state["u"]
        grads = self.diff.grad(u)
        # Burgers convection: u * ∂x u + vy * ∂y u
        conv = u * grads[0] + self.vy * grads[1]
        diffu = self.nu * self.diff.laplacian(u)
        return {"u": -conv + diffu}


def main() -> None:
    # 2D periodic domain
    grid = Grid.regular([(0.0, 2 * np.pi), (0.0, 2 * np.pi)], [32, 32], [True, True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = Burgers2D(grid, diff, nu=0.1, vy=0.5)
    # initial condition: sinusoidal wave
    ic = {"u": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1, 1], "phase": 0.0}}
    sim = Simulation(model, t0=0.0, t1=0.5, dt0=0.01, save_every=10, initial_state_params=ic)
    result = sim.run()
    # plot final state
    u0 = result.states[0]["u"]
    uf = result.states[-1]["u"]
    x = grid.coords[0]
    y = grid.coords[1]
    X, Y = np.meshgrid(x, y, indexing='ij')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(u0.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Initial u(x,y)")
    im1 = axes[1].imshow(uf.T, origin='lower', extent=(x[0], x[-1], y[0], y[-1]), aspect='auto')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("Final u(x,y)")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()