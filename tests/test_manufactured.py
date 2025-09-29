"""Manufactured solution tests for models.

These tests use analytic functions to verify that the discrete operators
produce the expected derivatives in the right‑hand sides of each model.
"""
import numpy as np

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator, FiniteDifference
from flexipde.models import LinearAdvection, Diffusion, ResistiveMHD, TwoFluid, IdealAlfven


def manufactured_cosine(grid):
    x = grid.coords[0]
    return np.cos(2.0 * x)


def test_advection_manufactured():
    # 1D periodic grid
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = LinearAdvection(grid, diff, velocity=[0.5])
    u0 = manufactured_cosine(grid)
    state = {"u": u0}
    rhs = model.rhs(state, t=0.0)
    # Analytical: ∂t u = -v * ∂x u = -0.5 * (-2 sin(2x)) = sin(2x)
    expected = np.sin(2.0 * grid.coords[0])
    assert np.allclose(rhs["u"], expected, atol=1e-8)


def test_diffusion_manufactured():
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = Diffusion(grid, diff, diffusivity=0.3)
    u0 = manufactured_cosine(grid)
    state = {"u": u0}
    rhs = model.rhs(state, t=0.0)
    # Analytical: ∂t u = D * ∇^2 u = 0.3 * (-4 cos(2x))
    expected = 0.3 * (-4.0) * np.cos(2.0 * grid.coords[0])
    assert np.allclose(rhs["u"], expected, atol=1e-8)


def test_resistive_mhd_manufactured():
    # 1D periodic grid
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = ResistiveMHD(grid, diff, eta=0.1)
    # v=B=cos(2x)
    u0 = manufactured_cosine(grid)
    state = {"v": u0, "B": u0}
    rhs = model.rhs(state, t=0.0)
    x = grid.coords[0]
    # dv/dt = dB/dx = -2 sin(2x)
    dv_expected = -2.0 * np.sin(2.0 * x)
    # dB/dt = dv/dx + eta ∇^2 B = -2 sin(2x) + 0.1 * (-4 cos(2x))
    dB_expected = -2.0 * np.sin(2.0 * x) + 0.1 * (-4.0) * np.cos(2.0 * x)
    assert np.allclose(rhs["v"], dv_expected, atol=1e-8)
    assert np.allclose(rhs["B"], dB_expected, atol=1e-8)


def test_two_fluid_manufactured():
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    velocities = ([0.5], [-1.0])
    model = TwoFluid(grid, diff, velocities=velocities)
    n_i0 = manufactured_cosine(grid)
    n_e0 = manufactured_cosine(grid)
    state = {"n_i": n_i0, "n_e": n_e0}
    rhs = model.rhs(state, t=0.0)
    x = grid.coords[0]
    # Ion: -0.5 * ∂x cos(2x) = -0.5 * (-2 sin(2x)) = sin(2x)
    # Electron: -(-1) * ∂x cos(2x) = 1 * (-2 sin(2x)) = -2 sin(2x)
    expected_i = np.sin(2.0 * x)
    expected_e = -2.0 * np.sin(2.0 * x)
    assert np.allclose(rhs["n_i"], expected_i, atol=1e-8)
    assert np.allclose(rhs["n_e"], expected_e, atol=1e-8)


def test_ideal_alfven_manufactured():
    grid = Grid.regular([(0.0, 2 * np.pi)], [64], [True])
    diff = SpectralDifferentiator(grid, backend="numpy")
    model = IdealAlfven(grid, diff)
    u0 = manufactured_cosine(grid)
    state = {"v": u0, "B": u0}
    rhs = model.rhs(state, t=0.0)
    x = grid.coords[0]
    dv_expected = -2.0 * np.sin(2.0 * x)
    dB_expected = -2.0 * np.sin(2.0 * x)
    assert np.allclose(rhs["v"], dv_expected, atol=1e-8)
    assert np.allclose(rhs["B"], dB_expected, atol=1e-8)