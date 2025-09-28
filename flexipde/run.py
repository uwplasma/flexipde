"""
Command line interface for flexipde.

This module can be executed with ``python -m flexipde.run`` to run a
simulation based on a configuration file.  The configuration must be
provided in TOML format; see :mod:`flexipde.io` for details.

Example::

    python -m flexipde.run config/mhd.toml --output results.npz

The script builds the simulation, runs it and saves the state arrays
at each saved time step into a compressed NumPy file if an output
filename is provided.  Otherwise it prints summary information to
stdout.
"""

from __future__ import annotations

import argparse
import pathlib
import numpy as _np

from flexipde.io import load_toml, build_simulation
from flexipde.result import SimulationResult

try:
    from rich.console import Console  # type: ignore
    _HAS_RICH = True
except ImportError:
    Console = None  # type: ignore
    _HAS_RICH = False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a flexipde simulation from a TOML configuration file.")
    parser.add_argument("config", type=str, help="Path to TOML configuration file")
    parser.add_argument("--output", type=str, help="Optional output filename to store results (will use gzip pickled format if the extension is .pkl.gz; otherwise .npz)")
    args = parser.parse_args(argv)
    cfg_path = pathlib.Path(args.config)
    cfg = load_toml(cfg_path)
    sim = build_simulation(cfg)
    # Display input summary
    console = Console() if _HAS_RICH else None
    def print_msg(msg: str) -> None:
        if console:
            console.print(msg)
        else:
            print(msg)
    model_name = cfg["model"]["type"] if "model" in cfg and "type" in cfg["model"] else sim.model.__class__.__name__
    discretisation = cfg.get("discretisation", {}).get("scheme", "spectral")
    solver_name = cfg.get("solver", {}).get("solver", sim.solver)
    ic_list = sim.initial_state_params_list
    num_runs = len(ic_list) if ic_list else 1
    print_msg(f"Starting simulation for model '{model_name}' with {num_runs} initial condition(s)")
    print_msg(f"  Grid shape: {sim.model.grid.shape}, discretisation: {discretisation}")
    print_msg(f"  Time interval: [{sim.t0}, {sim.t1}], solver: {solver_name}")
    import time as _time
    start_time = _time.perf_counter()
    results = sim.run()
    end_time = _time.perf_counter()
    elapsed = end_time - start_time
    # Normalize results to list
    if isinstance(results, tuple):
        results_list = [results]
    else:
        results_list = results
    # Save or print summary for each run
    if args.output:
        # Determine extension
        ext = pathlib.Path(args.output).suffix.lower()
        base = args.output
        # If multiple runs and base has no placeholder, append index
        for idx, (times, states) in enumerate(results_list):
            # Build SimulationResult
            initial_params = None
            if sim.initial_state_params_list:
                initial_params = sim.initial_state_params_list[idx]
            res_obj = SimulationResult(
                model_name=model_name,
                grid_cfg=sim.model.grid.to_config(),
                discretisation=discretisation,
                solver_name=solver_name if isinstance(solver_name, str) else solver_name.__class__.__name__,
                t0=sim.t0,
                t1=sim.t1,
                initial_params=initial_params,
                times=times,
                states=states,
                additional_info={}
            )
            # Determine filename per run
            if len(results_list) > 1:
                # Insert index before suffix
                out_path = pathlib.Path(base)
                stem = out_path.stem
                suffix = out_path.suffix
                filename = f"{stem}_{idx}{suffix}"
            else:
                filename = base
            if filename.endswith(".pkl.gz"):
                res_obj.save(filename)
            else:
                # save as npz: flatten states
                out = {}
                if states:
                    fieldnames = list(states[0].keys())
                    for name in fieldnames:
                        out[name] = _np.stack([s[name] for s in states], axis=0)
                    out['times'] = times
                    _np.savez_compressed(filename, **out)
                else:
                    _np.savez_compressed(filename, times=_np.array([]))
            print_msg(f"Results saved to {filename}")
    else:
        # Print summary to stdout
        for run_idx, (times, states) in enumerate(results_list):
            label = f"Run {run_idx}" if len(results_list) > 1 else "Run"
            print_msg(f"{label}: saved {len(times)} time point(s)")
            for i, t in enumerate(times):
                print_msg(f"  t = {t:.6g}")
                for name, arr in states[i].items():
                    print_msg(f"    {name}: min {arr.min():.6g}, max {arr.max():.6g}")
        print_msg(f"Simulation(s) completed in {elapsed:.3f} seconds")


if __name__ == "__main__":  # pragma: no cover
    main()
