import numpy as np
from flexipde.grid import Grid
from flexipde.discretisation import FiniteDifference
from flexipde.models.advection import LinearAdvection


def test_advection_rhs_consistency():
    # 1D periodic domain
    n = 32
    grid = Grid.regular([(0.0, 1.0)], [n], periodic=[True])
    diff = FiniteDifference(grid, backend="numpy")
    model = LinearAdvection(grid, diff, velocity=[1.0])
    state = model.initial_state()
    # compute rhs via model
    rhs = model.rhs(state, t=0.0)
    u = state["u"]
    du_dt = rhs["u"]
    # compute finite difference derivative manually
    dx = 1.0 / n
    # central difference with periodic wrap
    u_forward = np.roll(u, -1)
    u_backward = np.roll(u, 1)
    grad_u = (u_forward - u_backward) / (2 * dx)
    expected = -1.0 * grad_u
    assert np.allclose(du_dt, expected, atol=1e-6)