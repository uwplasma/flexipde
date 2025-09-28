import numpy as np
import pytest

def test_simulate_and_grad_skip_if_no_jax():
    """Ensure that simulate_and_grad raises RuntimeError when JAX/diffrax are missing."""
    import importlib
    has_jax = importlib.util.find_spec("jax") is not None
    has_diffrax = importlib.util.find_spec("diffrax") is not None
    if not (has_jax and has_diffrax):
        # Import inside assert to avoid NameError when missing
        from flexipde.optim import simulate_and_grad
        import flexipde
        # Build a dummy simulation that should not run
        grid = flexipde.Grid.regular([(0.0, 1.0)], [4], periodic=[True])
        diff = flexipde.discretisation.SpectralDifferentiator(grid, backend="numpy")
        model = flexipde.models.advection.LinearAdvection(grid, diff, velocity=[1.0])
        sim = flexipde.Simulation(model, t0=0.0, t1=0.1, dt0=0.01)
        with pytest.raises(RuntimeError):
            simulate_and_grad(sim, 1.0, lambda p: {"amplitude": p}, lambda s: np.array(0.0))
        return

@pytest.mark.skipif(
    not (pytest.importorskip("jax", reason="JAX not installed") and pytest.importorskip("diffrax", reason="diffrax not installed")),
    reason="JAX and diffrax required for this test",
)
def test_simulate_and_grad_advection():
    """Test gradient computation for a simple advection model using simulate_and_grad."""
    import jax
    import jax.numpy as jnp
    from flexipde.grid import Grid
    from flexipde.discretisation import SpectralDifferentiator
    from flexipde.models.advection import LinearAdvection
    from flexipde.solver import Simulation
    from flexipde.optim import simulate_and_grad

    grid = Grid.regular([(0.0, 2 * np.pi)], [32], periodic=[True])
    diff = SpectralDifferentiator(grid, backend="jax")
    model = LinearAdvection(grid, diff, velocity=[1.0])
    sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.1)
    def ic_from_params(p):
        return {"type": "sinusoidal", "amplitude": p, "wavevector": [1], "phase": 0.0, "backend": "jax"}
    def objective_fn(final_state):
        # Use mean of the field as a scalar objective
        u = final_state["u"]
        return jnp.mean(u)
    amp = jnp.array(1.0)
    loss, grad = simulate_and_grad(sim, amp, ic_from_params, objective_fn)
    assert loss.shape == ()
    assert grad.shape == ()