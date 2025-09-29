"""Configuration and I/O utilities.

This module provides functions for loading simulation setups from TOML
configuration files.  The configuration format is documented in the
``docs/user_guide.md``.
"""
from __future__ import annotations

from typing import Any, Dict, List

import os
import pathlib

try:
    import tomllib  # Python 3.11+
except Exception:
    import tomli as tomllib  # type: ignore

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
from .solver import Simulation


def load_toml(path: str) -> Dict[str, Any]:
    """Load a TOML file into a dictionary.

    If ``path`` does not end with ``.toml``, it is appended.  Relative
    paths are resolved relative to the current working directory.
    """
    p = pathlib.Path(path)
    if p.suffix != ".toml":
        p = p.with_suffix(".toml")
    with open(p, "rb") as f:
        return tomllib.load(f)


def build_simulation(cfg: Dict[str, Any] | str) -> Simulation:
    """Construct a :class:`Simulation` from a configuration.

    Parameters
    ----------
    cfg:
        A dictionary parsed from TOML or a path to a TOML file.

    Returns
    -------
    Simulation
        The configured simulation.
    """
    if isinstance(cfg, str):
        cfg = load_toml(cfg)
    # Grid
    gcfg = cfg.get("grid", {})
    domain = [(float(a), float(b)) for a, b in gcfg.get("domain", [])]
    shape = [int(n) for n in gcfg.get("shape", [])]
    periodic = [bool(p) for p in gcfg.get("periodic", [])]
    grid = Grid.regular(domain, shape, periodic)
    # Discretisation
    dcfg = cfg.get("discretisation", {})
    dtype = dcfg.get("type", "spectral").lower()
    backend = dcfg.get("backend", "numpy")
    if dtype == "spectral":
        diff = SpectralDifferentiator(grid, backend=backend)
    elif dtype in {"finite_difference", "fd"}:
        diff = FiniteDifference(grid, backend=backend)
    else:
        raise ValueError(f"Unknown discretisation type: {dtype}")
    # Model
    mcfg = cfg.get("model", {})
    mtype = mcfg.get("type", "advection").lower()
    # Extract model parameters: support both a nested ``parameters`` table and
    # parameters specified at the top level of the ``model`` section.
    mparams = dict(mcfg)
    mparams.pop("type", None)
    # If a [model.parameters] table was provided, update with its contents
    mparams.update(mcfg.get("parameters", {}))
    if mtype in {"advection", "linearadvection"}:
        model = LinearAdvection(grid, diff, velocity=mparams.get("velocity", [1.0] * grid.ndim))
    elif mtype in {"diffusion"}:
        model = Diffusion(grid, diff, diffusivity=float(mparams.get("diffusivity", 1.0)))
    elif mtype in {"resistive_mhd", "resistivemhd"}:
        model = ResistiveMHD(grid, diff, eta=float(mparams.get("eta", 0.01)))
    elif mtype in {"two_fluid", "twofluid"}:
        # velocities is list of list
        velocities = mparams.get("velocities", [[1.0] * grid.ndim, [-1.0] * grid.ndim])
        model = TwoFluid(grid, diff, velocities=velocities)
    elif mtype in {"drift_kinetic", "driftkinetic"}:
        model = DriftKinetic(grid, diff,
                             nv=int(mparams.get("nv", 32)),
                             v_min=float(mparams.get("v_min", -5.0)),
                             v_max=float(mparams.get("v_max", 5.0)),
                             E=float(mparams.get("E", 0.0)))
    elif mtype in {"alfven", "ideal_alfven"}:
        model = IdealAlfven(grid, diff)
    elif mtype in {"vlasov", "vlasov_two_stream", "two_stream"}:
        model = VlasovTwoStream(
            grid, diff,
            nv=int(mparams.get("nv", 64)),
            v_min=float(mparams.get("v_min", -5.0)),
            v_max=float(mparams.get("v_max", 5.0)),
            amplitude=float(mparams.get("amplitude", 0.05)),
            drift_velocity=float(mparams.get("drift_velocity", 2.0)),
            thermal_velocity=float(mparams.get("thermal_velocity", 1.0)),
            background_density=float(mparams.get("background_density", 1.0)),
        )
    else:
        raise ValueError(f"Unknown model type: {mtype}")
    # Simulation parameters
    scfg = cfg.get("simulation", {})
    t0 = float(scfg.get("t0", 0.0))
    t1 = float(scfg.get("t1", 1.0))
    dt0 = float(scfg.get("dt0", 0.01))
    save_every = int(scfg.get("save_every", 1))
    # Initial conditions
    icfg = cfg.get("initial_conditions", {})
    # Build simulation
    sim = Simulation(model, t0=t0, t1=t1, dt0=dt0, save_every=save_every,
                     initial_state_params=icfg)
    return sim