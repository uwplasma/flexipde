import numpy as np
import tempfile
import os

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.advection import LinearAdvection
from flexipde.models.diffusion import Diffusion
from flexipde.solver import Simulation
from flexipde.result import SimulationResult


def test_custom_initial_conditions_types():
    """Verify that custom initial conditions generate expected shapes and amplitudes."""
    grid = Grid.regular([(0.0, 2 * np.pi)], [32], periodic=[True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    # Sinusoidal with amplitude 2
    model = LinearAdvection(grid, diff, velocity=[1.0])
    ic = {"type": "sinusoidal", "amplitude": 2.0, "wavevector": [1], "phase": 0.0}
    state = model.initial_state(ic)
    u = state["u"]
    # Should be 2*sin(x)
    x = grid.coordinate_arrays("numpy")[0]
    assert np.allclose(u, 2.0 * np.sin(x), atol=1e-6)
    # Gaussian amplitude test
    ic2 = {"type": "gaussian", "amplitude": 3.0}
    state2 = model.initial_state(ic2)
    u2 = state2["u"]
    # Peak amplitude ~3
    assert np.isclose(u2.max(), 3.0, atol=1e-1)
    # Constant
    ic3 = {"type": "constant", "value": 5.0}
    state3 = model.initial_state(ic3)
    u3 = state3["u"]
    assert np.allclose(u3, 5.0)


def test_run_multiple_initial_conditions():
    """Check that Simulation can run multiple initial conditions and return a list of results."""
    grid = Grid.regular([(0.0, 2 * np.pi)], [32], periodic=[True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = Diffusion(grid, diff, diffusivity=0.0)  # zero diffusivity: constant in time
    # Two initial conditions: constant 1 and constant 2
    sim = Simulation(model, t0=0.0, t1=0.1, dt0=0.01)
    sim.initial_state_params_list = [
        {"type": "constant", "value": 1.0},
        {"type": "constant", "value": 2.0},
    ]
    results = sim.run()
    # Should return a list of two runs
    assert isinstance(results, list)
    assert len(results) == 2
    # Each result contains two time points (start and end)
    for times, states in results:
        assert len(times) >= 2
        assert len(states) == len(times)
        # Check that constant field remains constant
        u0 = states[0]["u"]
        u1 = states[-1]["u"]
        assert np.allclose(u0, u1)


def test_simulationresult_save_load(tmp_path):
    """Test saving and loading of SimulationResult."""
    # Create dummy result
    times = np.array([0.0, 1.0])
    states = [
        {"u": np.array([0.0, 1.0])},
        {"u": np.array([1.0, 2.0])},
    ]
    res = SimulationResult(
        model_name="test",
        grid_cfg={"domain": [[0, 1]], "shape": [2], "periodic": [True]},
        discretisation="spectral",
        solver_name="Euler",
        t0=0.0,
        t1=1.0,
        initial_params={"type": "constant", "value": 0},
        times=times,
        states=states,
    )
    filename = tmp_path / "result.pkl.gz"
    res.save(str(filename))
    loaded = SimulationResult.load(str(filename))
    assert loaded.model_name == res.model_name
    assert np.allclose(loaded.times, res.times)
    assert loaded.states[0]["u"][0] == res.states[0]["u"][0]