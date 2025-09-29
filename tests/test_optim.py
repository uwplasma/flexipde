"""Tests for optimisation helpers."""
import numpy as np
import pytest

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import LinearAdvection
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad


@pytest.mark.skipif(
    pytest.importorskip("jax", reason="JAX not installed") is None or
    pytest.importorskip("diffrax", reason="diffrax not installed") is None,
    reason="JAX and diffrax required"
)
def test_simulate_and_grad_simple():
    import jax.numpy as jnp  # type: ignore
    # simple advection with amplitude parameter
    grid = Grid.regular([(0.0, 2.0 * np.pi)], [32], [True])
    diff = SpectralDifferentiator(grid, backend="jax")
    model = LinearAdvection(grid, diff, velocity=[1.0])
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.05)

    def ic_from_params(p):
        return {"u": {"type": "sinusoidal", "amplitude": p, "wavevector": [1]}}

    def objective_fn(final_state):
        u = final_state["u"]
        # objective: mean squared value
        return jnp.mean(u * u)

    p = jnp.array(1.0)
    loss, grad = simulate_and_grad(sim, p, ic_from_params, objective_fn)
    assert np.isfinite(loss)
    assert np.isfinite(np.array(grad))