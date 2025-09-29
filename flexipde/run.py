"""Command‑line entry point for flexipde.

This module defines a ``main`` function that can be invoked via the
``flexipde`` console script.  It loads a configuration file, runs the
simulation, prints a summary and optionally plots the results.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, List, Dict

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
    _HAS_MPL = True
except Exception:  # pragma: no cover
    _HAS_MPL = False

from .io import build_simulation
from .solver import SimulationResult


def _summarise(result: SimulationResult) -> None:
    print(f"Simulation '{result.metadata['model']}' completed.")
    print(f"  t0 = {result.metadata['t0']}, t1 = {result.metadata['t1']}")
    fields = list(result.states[0].keys())
    print(f"  Fields: {fields}")
    # print final statistics
    last_state = result.states[-1]
    for name, arr in last_state.items():
        import numpy as np
        print(f"  {name}: mean={np.mean(arr):.4g}, max={np.max(arr):.4g}, min={np.min(arr):.4g}")


def _plot(result: SimulationResult) -> None:
    if not _HAS_MPL:
        print("matplotlib is not available; skipping plot.")
        return
    # Plot the final state for each field.  If multi‑dimensional, plot a slice.
    last_state = result.states[-1]
    fields = list(last_state.keys())
    import numpy as np
    nrows = 1
    ncols = len(fields)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3))
    if ncols == 1:
        axes = [axes]
    for ax, name in zip(axes, fields):
        data = last_state[name]
        # Determine dimension
        if data.ndim == 1:
            # 1D: line plot
            x = result.metadata.get("x", None)
            if x is None:
                x = range(len(data))
            ax.plot(x, data)
            ax.set_xlabel('x index')
            ax.set_ylabel(name)
        elif data.ndim == 2:
            # 2D: image
            im = ax.imshow(data.T, origin='lower', aspect='auto', interpolation='nearest')
            fig.colorbar(im, ax=ax)
            ax.set_xlabel('x index')
            ax.set_ylabel('v index')
        else:
            # high dimensional: plot mean over last axes
            mean = data.mean(axis=tuple(range(1, data.ndim)))
            ax.plot(mean)
            ax.set_ylabel(f"mean {name}")
            ax.set_xlabel('x index')
        ax.set_title(name)
    plt.tight_layout()
    plt.show()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a flexipde simulation from a TOML configuration file.")
    parser.add_argument('config', help="Path to the configuration file (without '.toml' suffix).", type=str)
    parser.add_argument('--plot', action='store_true', help="Plot the final state after running.")
    args = parser.parse_args(argv)
    sim = build_simulation(args.config)
    result_or_list = sim.run()
    if isinstance(result_or_list, list):
        for result in result_or_list:
            _summarise(result)
            if args.plot:
                _plot(result)
    else:
        result = result_or_list
        _summarise(result)
        if args.plot:
            _plot(result)

if __name__ == '__main__':  # pragma: no cover
    main()