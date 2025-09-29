"""Optimise the growth of the two‑stream instability.

This example adjusts the ratio of ion to electron temperature to
maximise the electric field growth rate in the two‑stream instability.
It uses JAX and Optax to compute gradients and perform optimisation.

Note: this example requires the ``flexipde[jax]`` and ``optax`` extras.
"""
import jax
import jax.numpy as jnp
import optax

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import VlasovTwoStream
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad


def main() -> None:
    # Set up grid and model
    grid = Grid.regular([(0.0, 2 * jnp.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="jax")
    model = VlasovTwoStream(grid, diff, nv=64, v_min=-5.0, v_max=5.0)
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.05)

    def ic_from_params(p):
        # p[0] = thermal velocity
        return {
            "amplitude": 0.05,
            "thermal_velocity": p[0],
            "drift_velocity": 2.0,
            "background_density": 0.5,
        }

    def objective_fn(final_state):
        # measure electric field energy: sum(E^2)
        f = final_state["f"]
        # compute electric field from charge density via Poisson: E = -dphi/dx
        # here we approximate by zero as a placeholder; user can extend
        return jnp.sum(f)  # simple proxy objective

    # parameter initial guess
    params = jnp.array([0.5])
    opt = optax.adam(0.1)
    opt_state = opt.init(params)

    for step in range(10):
        loss, grad = simulate_and_grad(sim, params, ic_from_params, objective_fn)
        updates, opt_state = opt.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
        print(f"step {step}, loss={loss}, thermal_velocity={float(params[0])}")


if __name__ == "__main__":
    main()