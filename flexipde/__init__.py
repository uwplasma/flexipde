"""Top‑level package for flexipde.

This module re‑exports the core classes and functions so that they can be
imported conveniently from :mod:`flexipde`.  For example::

    from flexipde import Grid, SpectralDifferentiator, Simulation

The full API is documented in the online docs.
"""
from .grid import Grid
from .discretisation import SpectralDifferentiator, FiniteDifference
from .models import (
    LinearAdvection,
    Diffusion,
    ResistiveMHD,
    TwoFluid,
    DriftKinetic,
    IdealAlfven,
    VlasovTwoStream,
)
from .solver import Simulation, SimulationResult
from .optim import simulate_and_grad, optimize_params
from .io import build_simulation

__all__ = [
    "Grid",
    "SpectralDifferentiator",
    "FiniteDifference",
    "LinearAdvection",
    "Diffusion",
    "ResistiveMHD",
    "TwoFluid",
    "DriftKinetic",
    "IdealAlfven",
    "VlasovTwoStream",
    "Simulation",
    "SimulationResult",
    "simulate_and_grad",
    "optimize_params",
    "build_simulation",
]