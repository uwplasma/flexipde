import numpy as np
from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.diffusion import Diffusion
from flexipde.solver import Simulation


def test_simulation_runs_with_euler():
    # Without JAX and Diffrax we expect to fall back to Euler integration
    n = 16
    grid = Grid.regular([(0.0, 1.0)], [n], periodic=[True])
    diff = FiniteDifference(grid, backend="numpy")
    model = Diffusion(grid, diff, diffusivity=0.1)
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.01, save_every=1)
    result = sim.run()
    # If multiple runs returned, take the first
    if isinstance(result, tuple):
        times, states = result
    else:
        times, states = result[0]
    # Should save at least two states (start and end)
    assert len(times) >= 2
    assert len(states) == len(times)
    # Check that solution remains real and finite
    for s in states:
        u = s['u']
        assert np.isfinite(u).all()