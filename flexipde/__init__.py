"""
flexipde
========

A flexible, high‑performance framework for plasma physics simulations.

This package provides a modular set of tools for defining and solving
systems of partial differential equations (PDEs) on structured grids using
JAX for automatic differentiation, just‑in‑time compilation and easy GPU
acceleration.  It aims to be simple to extend: users can add new
equations, boundary conditions, discretisation schemes or coordinate
systems without modifying the core solver.  The design is influenced by
research codes such as BOUT++ and Φ‑Flow, which emphasise modularity and
flexibility.  BOUT++ allows users to evolve any number of equations in
curvilinear geometry with a variety of numerical methods【501252340464299†L24-L33】.  Φ‑Flow provides
object‑oriented, reusable simulations that run on a variety of backends
(NumPy, PyTorch, TensorFlow, JAX) and dimensions【383486560010699†L364-L376】.  flexipde
builds on these ideas, combining a declarative configuration format with
the performance of JAX and the ergonomics of Equinox and Diffrax.

The top‑level API exposes a few classes that are most commonly used:

* :class:`flexipde.grid.Grid` – encapsulates domain geometry and metric
  information.
* :class:`flexipde.models.base.PDEModel` – base class for PDE models.
* :class:`flexipde.solver.Simulation` – orchestrates time integration
  using Diffrax and handles checkpointing and output.

See the :mod:`flexipde.examples` module for working examples.
"""

from importlib import metadata as _metadata

from .grid import Grid
from .solver import Simulation
from .models.base import PDEModel
from .optim import simulate_and_grad, optimize_params

__all__ = [
    "Grid",
    "Simulation",
    "PDEModel",
    # Optimisation helpers
    "simulate_and_grad",
    "optimize_params",
]

try:
    # Prefer version information from package metadata
    __version__ = _metadata.version(__name__)
except Exception:
    __version__ = "0.1.0"