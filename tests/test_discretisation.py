"""Tests for spatial discretisation schemes."""
import numpy as np
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator, FiniteDifference


def test_spectral_grad_and_laplacian_1d():
    # 1D grid
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    x = grid.coords[0]
    u = np.cos(2 * x)
    # gradient of cos(2x) = -2 sin(2x)
    grad = diff.grad(u)[0]
    expected_grad = -2.0 * np.sin(2 * x)
    assert np.allclose(grad, expected_grad, atol=1e-8)
    # laplacian of cos(2x) = -4 cos(2x)
    lap = diff.laplacian(u)
    expected_lap = -4.0 * np.cos(2 * x)
    assert np.allclose(lap, expected_lap, atol=1e-8)


def test_finite_difference_grad_1d():
    grid = Grid.regular([(0.0, 1.0)], [101], [False])
    diff = FiniteDifference(grid, backend="numpy")
    x = grid.coords[0]
    u = x ** 2
    # gradient should be 2x
    grad = diff.grad(u)[0]
    expected = 2.0 * x
    assert np.allclose(grad[1:-1], expected[1:-1], atol=1e-2)