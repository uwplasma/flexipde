"""Time integration driver.

The :mod:`flexipde.solver` module provides the :class:`Simulation`
class for integrating models in time.  It automatically uses JAX and
Diffrax when available and appropriate, but falls back to an explicit
Euler integrator when JAX is not installed or when the discretiser
backend is NumPy.  Simulations can run a single initial condition or
multiple initial conditions in sequence.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt  # type: ignore[import]
    _JAX_AVAILABLE = True
except Exception:  # pragma: no cover
    jax = None
    jnp = None
    _JAX_AVAILABLE = False

from .models.base import PDEModel
from .discretisation.spectral import SpectralDifferentiator


@dataclass
class SimulationResult:
    """Container for simulation output.

    Stores the time array and the list of state dictionaries at each saved
    time.  Additional metadata such as the model name and parameters are
    also included.
    """
    times: _np.ndarray
    states: List[Dict[str, Any]]
    metadata: Dict[str, Any]

    def save(self, filename: str) -> None:
        """Save the result to a compressed NumPy ``.npz`` file."""
        data = {"times": self.times}
        for i, state in enumerate(self.states):
            for k, v in state.items():
                data[f"{k}_{i}"] = v
        _np.savez_compressed(filename, **data)

    # Make the result unpackable like (times, states) for backward compatibility
    def __iter__(self):
        """Iterate over ``(times, states)`` to allow unpacking.

        Examples
        --------
        >>> result = SimulationResult(...)
        >>> times, states = result
        """
        return iter((self.times, self.states))

    def __getitem__(self, index: int):  # pragma: no cover
        """Allow indexing for backward compatibility.

        ``result[0]`` returns ``times``; ``result[1]`` returns ``states``.
        Any other index raises ``IndexError``.
        """
        if index == 0:
            return self.times
        elif index == 1:
            return self.states
        raise IndexError("SimulationResult only supports indices 0 (times) and 1 (states)")


@dataclass
class Simulation:
    """A time integration wrapper for PDE models.

    Parameters
    ----------
    model:
        The PDEModel instance.
    t0, t1:
        Start and end times.
    dt0:
        Initial time step.  For the Euler fallback, this is the fixed
        step size.  For Diffrax it is the initial guess.
    save_every:
        How often to save the solution when using the fallback integrator.
        Ignored when using Diffrax; the solution is saved only at t0 and t1.
    """
    model: PDEModel
    t0: float
    t1: float
    dt0: float
    save_every: int = 1
    initial_state_params: Optional[Dict[str, Dict[str, Any]]] = None
    initial_state_params_list: Optional[List[Dict[str, Dict[str, Any]]]] = None

    def run(self) -> SimulationResult | List[SimulationResult]:
        """Run the simulation.

        If ``initial_state_params_list`` is provided, runs the simulation
        for each set of parameters and returns a list of results.  Otherwise
        runs a single simulation.
        """
        if self.initial_state_params_list is not None:
            results = []
            for ic in self.initial_state_params_list:
                old = self.initial_state_params
                self.initial_state_params = ic
                res = self._run_single()
                results.append(res)
            # restore
            self.initial_state_params = None
            return results
        else:
            return self._run_single()

    def _run_single(self) -> SimulationResult:
        # build initial state
        state0 = self.model.initial_state(self.initial_state_params)
        # Determine whether to use JAX/diffrax
        use_jax = False
        # Use JAX if available, model.diff uses JAX backend, and diffrax is importable
        if _JAX_AVAILABLE and isinstance(self.model.diff, SpectralDifferentiator):
            if getattr(self.model.diff, "_backend", "numpy") == "jax":
                use_jax = True
        # Always treat non‑linear models as using JAX for gradient computing if available
        times: List[float] = []
        states: List[Dict[str, Any]] = []
        times.append(self.t0)
        states.append(state0)
        if use_jax:
            # JAX integration with diffrax.  Avoid converting JAX arrays
            # to NumPy inside the differentiable portion of the code.
            y0 = state0  # state0 is already a dictionary of JAX arrays
            model = self.model

            def rhs_fn(t, y, args):
                # t is a JAX scalar; do not cast to float.  Compute the
                # derivative using the model's rhs, which returns a dict
                # of JAX arrays.
                return model.rhs(y, t)

            term = ODETerm(rhs_fn)
            solver = Dopri5()
            # Save only the initial and final states.  diffrax will
            # return a solution object whose ``ys`` attribute is the
            # final state at t1.
            saveat = SaveAt(t0=True, t1=True)
            sol = diffeqsolve(
                term,
                solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=y0,
                saveat=saveat,
            )
            # times remains a list of floats for compatibility
            times = [self.t0, self.t1]
            # states is a list of dictionaries: the initial state and
            # the final state returned by diffrax.  Do not convert
            # JAX arrays to NumPy; leave them as JAX arrays so that
            # gradients can flow through them.
            states = [state0, sol.ys]
        else:
            # explicit Euler fallback
            t = self.t0
            state = state0
            step = 0
            dt = self.dt0
            while t < self.t1 - 1e-12:
                # compute rhs
                dy = self.model.rhs(self.model.apply_bcs(state), t)
                # update
                state = {k: v + dt * dy[k] for k, v in state.items()}
                t = t + dt
                step += 1
                if step % self.save_every == 0:
                    times.append(t)
                    states.append(state)
            # ensure final state saved
            if times[-1] < self.t1 - 1e-12:
                # final small step
                dt_last = self.t1 - t
                dy = self.model.rhs(self.model.apply_bcs(state), t)
                state = {k: v + dt_last * dy[k] for k, v in state.items()}
                t = self.t1
                times.append(t)
                states.append(state)
        # build result
        meta = {
            "model": self.model.__class__.__name__,
            "t0": self.t0,
            "t1": self.t1,
            "dt0": self.dt0,
        }
        return SimulationResult(times=_np.array(times), states=states, metadata=meta)