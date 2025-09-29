"""Optimise a transport equation parameter.

This script demonstrates how to optimise a parameter in a transport
equation such that the solution reaches a steady state.  A scalar field
obeys ``\partial_t u + \lambda\,\partial_x u = 0`` with periodic
boundaries.  We adjust ``\lambda`` to minimise the variance of the
solution over time, effectively searching for a velocity that yields a
stationary profile relative to the initial condition.

Requires the ``flexipde[jax]`` and ``optax`` extras.
"""
import jax
import jax.numpy as jnp
import optax

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import LinearAdvection
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad


def main() -> None:
    grid = Grid.regular([(0.0, 2 * jnp.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="jax")
    # velocity will be supplied via parameter

    def build_model(lam):
        return LinearAdvection(grid, diff, velocity=[lam])

    def ic_from_param(p):
        # initial sinusoid independent of parameter
        return {"u": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0, "backend": "jax"}}

    def objective_fn(final_state):
        u = final_state["u"]
        return jnp.var(u)  # variance: small if u is flat

    param = jnp.array([1.0])
    opt = optax.adam(0.2)
    opt_state = opt.init(param)

    for step in range(20):
        # build simulation with current parameter
        model = build_model(float(param[0]))
        sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.05)
        # compute loss and gradient with respect to param
        loss, grad = simulate_and_grad(sim, param, ic_from_param, objective_fn)
        updates, opt_state = opt.update(grad, opt_state)
        param = optax.apply_updates(param, updates)
        print(f"step {step}, loss={loss}, lambda={float(param[0])}")


if __name__ == "__main__":
    main()