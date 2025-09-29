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

# -----------------------------------------------------------------------------
# Helper functions for safely evaluating numeric expressions in configuration
# files.  TOML does not permit arbitrary expressions like ``2*pi`` or
# scientific notation outside of numbers.  To allow users to specify
# constants such as ``"2*pi"`` in the ``domain`` or ``dimensions`` fields,
# we provide a minimal arithmetic evaluator.  Only simple arithmetic
# expressions with addition, subtraction, multiplication, division and
# exponentiation are permitted, and the constants ``pi``, ``tau`` and ``e``
# are allowed.  See ``docs/installation.md`` for details.
import ast
import operator as _op
import math as _math

_ALLOWED_OPS: Dict[Any, Any] = {
    ast.Add: _op.add,
    ast.Sub: _op.sub,
    ast.Mult: _op.mul,
    ast.Div: _op.truediv,
    ast.Pow: _op.pow,
    ast.USub: _op.neg,
    ast.UAdd: _op.pos,
}
_CONST_MAP: Dict[str, float] = {
    "pi": _math.pi,
    "tau": 2 * _math.pi,
    "e": _math.e,
}


def _safe_eval(expr: str) -> float:
    """Safely evaluate a simple arithmetic expression to a float.

    Only expressions consisting of numbers, the constants ``pi``, ``tau`` and
    ``e``, and the operators +, -, *, /, and ** are allowed.  Anything else
    raises ``ValueError``.  This function is used to interpret strings like
    ``"2*pi"`` or ``"1e-6*1e-6"`` found in configuration files.

    Parameters
    ----------
    expr:
        The expression to evaluate.

    Returns
    -------
    float
        The evaluated numeric value.

    Raises
    ------
    ValueError
        If the expression contains unsupported operations or names.
    """
    node = ast.parse(expr, mode="eval").body

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Num):  # type: ignore[attr-defined]
            return n.n  # type: ignore[attr-defined]
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in _ALLOWED_OPS:
            left = _eval(n.left)  # type: ignore[arg-type]
            right = _eval(n.right)  # type: ignore[arg-type]
            return _ALLOWED_OPS[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _ALLOWED_OPS:
            return _ALLOWED_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONST_MAP:
            return _CONST_MAP[n.id]
        raise ValueError(f"Unsafe or unsupported expression: {expr}")

    return float(_eval(node))


def _coerce_dims(dims: Iterable[Iterable[Any]]) -> List[Tuple[float, float]]:
    """Coerce a sequence of dimension pairs to floats via safe evaluation.

    Each element ``(a, b)`` may be a number or a string expression.  This
    helper evaluates any string using :func:`_safe_eval` and returns a list
    of ``(float(a), float(b))`` pairs.  Non‑string values are converted
    directly to float.
    """
    out: List[Tuple[float, float]] = []
    for pair in dims:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("Domain and dimensions must be sequences of pairs")
        a, b = pair
        # If a or b are strings, evaluate them safely.  Otherwise cast to float.
        if isinstance(a, str):  # type: ignore
            a_val = _safe_eval(a)
        else:
            a_val = float(a)
        if isinstance(b, str):  # type: ignore
            b_val = _safe_eval(b)
        else:
            b_val = float(b)
        out.append((float(a_val), float(b_val)))
    return out


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
    # Support both legacy keys (domain, shape) and new keys (dimensions, resolution)
    if "dimensions" in gcfg:
        raw_dims = gcfg.get("dimensions", [])
        domain = _coerce_dims(raw_dims)
    else:
        raw_dom = gcfg.get("domain", [])
        domain = _coerce_dims(raw_dom)
    if "resolution" in gcfg:
        shape = [int(n) for n in gcfg.get("resolution", [])]
    else:
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
    mparams = mcfg.get("parameters", {})
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
        # Optional B0 parameter for ideal Alfvén model
        model = IdealAlfven(grid, diff, B0=float(mparams.get("B0", 1.0)))
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