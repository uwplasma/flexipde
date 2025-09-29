"""Tests for initial condition handling and multi‑run functionality."""
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import Diffusion
from flexipde.solver import Simulation


def test_run_multiple_initial_conditions():
    # two initial constant values; diffusion with zero diffusivity should keep constant in time
    grid = Grid.regular([(0.0, 1.0)], [32], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = Diffusion(grid, diff, diffusivity=0.0)
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.01)
    sim.initial_state_params_list = [
        {"u": {"type": "constant", "value": 1.0}},
        {"u": {"type": "constant", "value": 2.0}},
    ]
    results = sim.run()
    assert isinstance(results, list)
    assert len(results) == 2
    # Check constancy
    for res, expected in zip(results, [1.0, 2.0]):
        final_u = res.states[-1]["u"]
        assert np.allclose(final_u, expected)