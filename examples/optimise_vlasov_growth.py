"""
Optimise the growth rate of the two–stream instability.

This script uses the optimisation utilities in :mod:`flexipde.optim`
to tune the thermal velocity of the initial distribution in the
Vlasov–Poisson model.  The objective is the negative of the
electric field energy at the end of the simulation; maximising
growth therefore corresponds to minimising this negative value.

Requirements
------------
This example requires the JAX and optimisation extras to be
installed::

    pip install flexipde[jax,optim]

It also requires Optax.
"""

from __future__ import annotations

import optax  # type: ignore[import-untyped]
import jax.numpy as jnp
from flexipde import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import VlasovTwoStream
from flexipde.solver import Simulation
from flexipde.optim import optimize_params


def main() -> None:
    # Build the simulation with a JAX backend
    grid = Grid.regular([(0.0, 2 * jnp.pi)], [128], periodic=[True])
    diff = SpectralDifferentiator(grid, backend="jax")
    model = VlasovTwoStream(grid, diff, nv=64, v_min=-5.0, v_max=5.0)
    sim = Simulation(model, t0=0.0, t1=10.0, dt0=0.1, solver="Dopri5")

    # Define parameterised initial conditions
    def ic_from_params(p):
        """Return initial condition parameters from the optimisation parameter.

        We avoid converting the JAX parameter ``p`` to a Python float here.  The
        VlasovTwoStream model is differentiable with respect to the thermal
        velocity when ``p`` is a JAX scalar.  Setting ``backend='jax'`` is
        unnecessary because the optimisation utilities will automatically
        enforce a JAX backend for differentiability.
        """
        return {
            "thermal_velocity": p,
            "amplitude": 1e-3,
        }

    # Objective: negative electric field energy at final time
    def objective_fn(final_state):
        f = final_state["f"]
        # Compute electric field using model’s Poisson solver and JAX backend
        E = model._poisson_field(f, jnp)
        return -jnp.mean(E ** 2)

    # Optimiser: use ADAM for gradient ascent
    optimiser = optax.adam(learning_rate=0.05)
    init_param = 1.0

    loss, optimal_param = optimize_params(
        sim,
        init_params=init_param,
        ic_from_params=ic_from_params,
        objective_fn=objective_fn,
        optimizer=optimiser,
        num_steps=20,
    )
    print(f"Optimal thermal_velocity: {float(optimal_param):.5f}\nFinal objective: {float(loss):.6e}")


if __name__ == "__main__":
    main()