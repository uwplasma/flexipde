"""
Input/output utilities.

This module provides functions to load configuration files and build
simulation objects.  Configuration files are written in TOML and
describe the grid, discretisation scheme, model parameters and solver
settings.  Users can run simulations from the command line by
providing a TOML file to the ``flexipde.run`` module.

Example TOML file
-----------------

    [grid]
    domain = [[0.0, 2.0], [0.0, 2.0]]
    shape = [128, 128]
    periodic = [true, true]

    [discretisation]
    scheme = "spectral"  # options: spectral, finite_difference
    backend = "jax"

    [model]
    type = "advection"
    velocity = [1.0, 0.0]

    [solver]
    t0 = 0.0
    t1 = 1.0
    dt0 = 0.01
    solver = "Dopri5"

After writing such a file, run ``python -m flexipde.run path/to/config.toml`` to
execute the simulation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Sequence

import pathlib
import importlib
import sys
import types

try:
    import tomllib  # Python >=3.11
except ImportError:
    import tomli as tomllib  # type: ignore

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator, FiniteDifference
from flexipde.models.advection import LinearAdvection
from flexipde.models.diffusion import Diffusion
from flexipde.models.ideal_mhd import IdealAlfven
from flexipde.solver import Simulation


def load_toml(path: str | pathlib.Path) -> Dict[str, Any]:
    """Load a TOML file and return the contents as a dict."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def build_simulation(config: Dict[str, Any]) -> Simulation:
    """Construct a :class:`Simulation` from a configuration dictionary."""
    # Grid
    grid_cfg = config.get("grid")
    if grid_cfg is None:
        raise ValueError("Configuration must define a [grid] section")
    grid = Grid.from_config(grid_cfg)
    # Discretisation
    disc_cfg = config.get("discretisation", {})
    scheme = disc_cfg.get("scheme", "spectral").lower()
    backend = disc_cfg.get("backend")
    if scheme == "spectral":
        diff = SpectralDifferentiator(grid, backend=backend)
    elif scheme in {"finite_difference", "fd"}:
        diff = FiniteDifference(grid, backend=backend)
    else:
        raise ValueError(f"Unknown discretisation scheme {scheme}")
    # Model
    model_cfg = config.get("model")
    if model_cfg is None:
        raise ValueError("Configuration must define a [model] section")
    model_type = model_cfg.get("type")
    if model_type is None:
        raise ValueError("Model section must specify a 'type'")
    model_type = model_type.lower()
    if model_type in {"advection", "linear_advection"}:
        velocity = model_cfg.get("velocity")
        if velocity is None:
            raise ValueError("Advection model requires a velocity vector")
        model = LinearAdvection(grid, diff, velocity=velocity)
    elif model_type in {"diffusion"}:
        diffusivity = model_cfg.get("diffusivity", 1.0)
        model = Diffusion(grid, diff, diffusivity=diffusivity)
    elif model_type in {"ideal_mhd", "alfven", "mhd"}:
        B0 = model_cfg.get("B0", [1.0])
        model = IdealAlfven(grid, diff, B0=B0)
    else:
        raise ValueError(f"Unknown model type {model_type}")
    # Solver
    solver_cfg = config.get("solver", {})
    t0 = solver_cfg.get("t0", 0.0)
    t1 = solver_cfg.get("t1", 1.0)
    dt0 = solver_cfg.get("dt0")
    solver_name = solver_cfg.get("solver", "Dopri5")
    save_every = solver_cfg.get("save_every")
    sim = Simulation(model, t0=t0, t1=t1, dt0=dt0, solver=solver_name,
                     save_every=save_every)
    # Handle initial conditions
    ic_cfg = config.get("initial_conditions")
    if ic_cfg is not None:
        # Accept either a list of dicts or a single dict
        ic_list: list[Dict[str, Any]] = []
        if isinstance(ic_cfg, list):
            for item in ic_cfg:
                if not isinstance(item, dict):
                    raise ValueError("Each entry in 'initial_conditions' must be a table/dict")
                ic_list.append(item)
        elif isinstance(ic_cfg, dict):
            ic_list.append(ic_cfg)
        else:
            raise ValueError("'initial_conditions' must be a table or list of tables")
        sim.initial_state_params_list = ic_list
    return sim
