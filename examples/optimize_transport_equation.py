"""Example: optimise a transport equation for equilibrium.

This script demonstrates how to search for a transport coefficient that
minimises the time variation of a 1D transport equation.  We define a
custom model and use gradient descent to adjust a parameter so that the
solution becomes as stationary as possible.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

import numpy as np
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.base import PDEModel
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad


class Transport1D(PDEModel):
    """Custom 1D transport equation with variable coefficient.

    The equation is \(\partial_t u + a \partial_x u = 0\) where `a`
    is a scalar parameter.  We treat `a` as part of the model so that
    gradients can be taken with respect to it.
    """

    def __init__(self, grid, diff, a: float):
        self.a = a
        super().__init__(grid=grid, diff=diff)

    def __post_init__(self):
        self.field_names = ["u"]
        self.field_bcs = ["periodic"] * self.grid.ndim
        super().__post_init__()

    def rhs(self, state, t):
        u = state["u"]
        grad_u = self.diff.grad(u)[0]
        return {"u": -self.a * grad_u}


def main() -> None:
    # Use JAX backend for differentiability
    grid = Grid.regular([(0.0, 2.0 * jnp.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="jax")

    # Initial amplitude
    ic = {"u": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0, "backend": "jax"}}

    # Objective: minimise the squared difference between initial and final state
    def objective_fn(final_state, initial_state):
        return jnp.sum((final_state["u"] - initial_state["u"]) ** 2)

    def loss_with_grad(a):
        model = Transport1D(grid, diff, a)
        sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.1)
        # We fix the initial condition for all runs
        def ic_fn(_): return ic
        def obj_fn(final_state):
            # closure capturing initial state: we pass through to objective
            # but jax value and grad cannot capture Python objects so we compute
            return objective_fn(final_state, sim._initial_state)
        # Use simulate_and_grad to get gradient of loss wrt a
        loss, grad = simulate_and_grad(sim, a, lambda p: ic, lambda st: objective_fn(st, sim._initial_state))
        return loss, grad

    a = jnp.array(1.0)
    opt = optax.adam(0.1)
    opt_state = opt.init(a)
    for step in range(10):
        loss, grad = loss_with_grad(a)
        updates, opt_state = opt.update(grad, opt_state)
        a = optax.apply_updates(a, updates)
        print(f"Step {step}: loss={loss:.4e}, a={float(a):.4f}")

    # Compare initial and final solution for the optimised parameter
    model = Transport1D(grid, diff, a)
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.1, initial_state_params=ic)
    result = sim.run()
    import matplotlib.pyplot as plt
    x = grid.coords[0]
    u0 = result.states[0]["u"]
    u_end = result.states[-1]["u"]
    plt.plot(x, np.array(u0), label="Initial")
    plt.plot(x, np.array(u_end), label="Final")
    plt.legend()
    plt.title("Optimised transport equation")
    plt.show()


if __name__ == "__main__":
    main()