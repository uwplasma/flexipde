import numpy as np
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator, FiniteDifference


def test_spectral_derivative_sine():
    # 1D periodic grid
    L = 2 * np.pi
    n = 64
    grid = Grid.regular([(0.0, L)], [n], periodic=[True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    x = grid.coordinates[0]
    u = np.sin(x)
    du_dx = diff.grad(u, axis=0)
    exact = np.cos(x)
    # Spectral derivative should be very accurate
    assert np.allclose(du_dx, exact, atol=1e-6)


def test_fd_derivative_sine():
    L = 2 * np.pi
    n = 64
    grid = Grid.regular([(0.0, L)], [n], periodic=[True])
    diff = FiniteDifference(grid, backend="numpy")
    x = grid.coordinates[0]
    u = np.sin(x)
    du_dx = diff.grad(u, axis=0)
    exact = np.cos(x)
    # Finite difference is second order; larger tolerance
    assert np.allclose(du_dx, exact, atol=1e-2)