"""Example: optimise two–stream instability growth rate.

This script demonstrates how to use gradient–based optimisation to find a
thermal velocity parameter that minimises the growth rate of the two–stream
instability.  We differentiate the simulation with respect to the thermal
velocity using JAX and Diffrax, and update the parameter using Optax.

To run this example you must install the JAX extras:

    pip install 'flexipde[jax]'

"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import VlasovTwoStream
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad


def main() -> None:
    # Define grid and model using JAX backend
    grid = Grid.regular([(0.0, 2.0 * jnp.pi)], [32], [True])
    diff = SpectralDifferentiator(grid, backend="jax")
    model = VlasovTwoStream(grid, diff, nv=64, v_min=-5.0, v_max=5.0)
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.05)

    def ic_from_param(p):
        return {
            "f": {
                "amplitude": 0.05,
                "drift_velocity": 1.0,
                "thermal_velocity": p,
                "background_density": 1.0,
                "backend": "jax",
            }
        }

    def objective_fn(final_state):
        # Negative of density fluctuation amplitude at final time as objective
        f_end = final_state["f"]
        dv = (model.v_max - model.v_min) / model.nv
        rho = jnp.sum(f_end, axis=1) * dv
        # return squared norm of density perturbation
        return jnp.sum((rho - 1.0) ** 2)

    # Optimise parameter using simple gradient descent
    param = jnp.array(1.0)
    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(param)
    for step in range(10):
        loss, grad = simulate_and_grad(sim, param, ic_from_param, objective_fn)
        updates, opt_state = opt.update(grad, opt_state)
        param = optax.apply_updates(param, updates)
        print(f"Step {step}: loss={loss:.4e}, param={float(param):.4f}")


if __name__ == "__main__":
    main()