"""
Simulation result container.

This module provides the :class:`SimulationResult` class which stores
both the input parameters and the simulation outputs.  It can be
serialised to a compressed file for later analysis and plotting.

The result object stores the model name, grid configuration,
discretisation scheme, solver settings, initial condition parameters,
and the times and states saved during the run.  Saving uses
Python's pickle protocol wrapped in gzip compression for compact
storage.  Users can reload a result using the :meth:`load` class
method.

Example
-------

>>> res = SimulationResult(model_name="advection", ...)
>>> res.save("run1.pkl.gz")
>>> res2 = SimulationResult.load("run1.pkl.gz")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Sequence, Optional

import numpy as np
import gzip
import pickle


@dataclass
class SimulationResult:
    """Container for simulation output and metadata.

    Parameters
    ----------
    model_name : str
        Name of the model used.
    grid_cfg : dict
        Configuration dictionary describing the grid.  Typically
        obtained from :meth:`flexipde.grid.Grid.to_config`.
    discretisation : str
        Name of the discretisation scheme used.
    solver_name : str
        Name of the solver used.
    t0, t1 : float
        Start and end times of the simulation.
    initial_params : dict
        Parameters used to generate the initial condition.
    times : ndarray
        Array of time points at which the state was saved.
    states : list of dict
        List of field dictionaries corresponding to saved times.
    additional_info : dict, optional
        Any additional metadata to store.
    """

    model_name: str
    grid_cfg: Dict[str, Any]
    discretisation: str
    solver_name: str
    t0: float
    t1: float
    initial_params: Optional[Dict[str, Any]]
    times: np.ndarray
    states: Sequence[Dict[str, Any]]
    additional_info: Optional[Dict[str, Any]] = None

    def save(self, filename: str) -> None:
        """Serialise this result to ``filename`` using gzip compression."""
        with gzip.open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str) -> "SimulationResult":
        """Load a result from a compressed file created with :meth:`save`."""
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)