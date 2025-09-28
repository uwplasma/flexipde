"""
Time integration of PDE models.

This module defines the :class:`Simulation` class which orchestrates
the solution of a PDE model in time.  It wraps the numerical
differential equation solvers provided by Diffrax, which is a JAX‑
based library for solving ODEs, SDEs and other controlled
differential equations.  Diffrax supports a wide variety of explicit
and implicit solvers and can operate on arbitrary PyTree states【805256974599970†L76-L91】.

The :class:`Simulation` takes a :class:`~flexipde.models.base.PDEModel`
and repeatedly calls its :meth:`rhs` to obtain time derivatives.
Boundary conditions are applied on the state at each step via
``apply_boundary_conditions``.  Users can customise the solver type,
initial step size, termination time and saving strategy.

Example
-------

>>> from flexipde.grid import Grid
>>> from flexipde.discretisation import SpectralDifferentiator
>>> from flexipde.models.advection import LinearAdvection
>>> from flexipde.solver import Simulation
>>> grid = Grid.regular([(0, 2*_np.pi)], [128], periodic=[True])
>>> diff = SpectralDifferentiator(grid)
>>> model = LinearAdvection(grid, diff, velocity=[1.0])
>>> sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.01)
>>> times, states = sim.run()

This will evolve a sine wave one full period along the domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Sequence, Optional, Tuple

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

try:
    from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, BacksolveAdjoint, SaveAt
    _HAS_DIFFRAX = True
except ImportError:
    diffeqsolve = None  # type: ignore
    ODETerm = None  # type: ignore
    Dopri5 = None  # type: ignore
    Tsit5 = None  # type: ignore
    BacksolveAdjoint = None  # type: ignore
    SaveAt = None  # type: ignore
    _HAS_DIFFRAX = False

from flexipde.models.base import PDEModel


@dataclass
class Simulation:
    """Simulation of a PDE model using Diffrax.

    Parameters
    ----------
    model : :class:`~flexipde.models.base.PDEModel`
        The PDE model to simulate.
    t0 : float
        Initial time.
    t1 : float
        Final time.
    dt0 : float, optional
        Initial step size for adaptive integrators.  If ``None``, an
        internal default is used by Diffrax.
    solver : str or solver instance, optional
        Name of the solver to use.  Recognised values include
        ``"Dopri5"`` and ``"Tsit5"`` for explicit Runge–Kutta
        methods, and any solver class defined in Diffrax.  If not
        specified, ``Dopri5`` is used.
    save_every : int, optional
        Save solution every ``save_every`` steps for the fallback
        integrator.  When using Diffrax, dense output is always
        obtained and saved at the end of the integration.
    save_at : callable, optional
        User defined callback determining whether to save the state.
    initial_state_params_list : Sequence[Dict[str, Any]], optional
        When provided, run a separate simulation for each set of
        initial condition parameters.  If more than one set is
        supplied and Diffrax with JAX is available, the solver will
        attempt to vectorise over the batch using ``jax.vmap``.  If
        this attribute is ``None`` (default) then a single run is
        executed using default initial conditions.
    """
    model: PDEModel
    t0: float
    t1: float
    dt0: Optional[float] = None
    solver: str | Any = "Dopri5"
    save_every: Optional[int] = None
    save_at: Optional[Callable[[float, Dict[str, Any], int], bool]] = None
    initial_state_params_list: Optional[Sequence[Dict[str, Any]]] = None

    def _choose_solver(self) -> Any:
        """Return a Diffrax solver instance or ``None``.

        If Diffrax is not available or JAX is not available, this
        method returns ``None`` so that the fallback integrator will be
        used.  Otherwise it returns an instance of the requested
        solver.  If the solver name is unrecognised, a ``ValueError``
        is raised.
        """
        if not _HAS_DIFFRAX or not _HAS_JAX:
            return None
        # Determine solver class from string name
        if isinstance(self.solver, str):
            name = self.solver.lower()
            if name == "dopri5":
                return Dopri5()
            elif name == "tsit5":
                return Tsit5()
            else:
                try:
                    cls = getattr(__import__("diffrax", fromlist=[self.solver]), self.solver)
                    return cls()
                except Exception as e:
                    raise ValueError(f"Unknown solver {self.solver}") from e
        # Otherwise treat as an instance or callable
        return self.solver

    def run(self):
        """Run the simulation or simulations depending on the configured
        initial conditions.

        If ``initial_state_params_list`` is provided and contains more
        than one entry, this method will return a list of results (one
        per initial condition).  Otherwise a single result is
        returned.  Each result is a tuple ``(times, states)`` with
        ``times`` an array of save times and ``states`` a list of
        state dictionaries.
        """
        if self.initial_state_params_list:
            if len(self.initial_state_params_list) == 1:
                ic = self.initial_state_params_list[0]
                return self._run_single(ic)
            else:
                return self._run_multiple(self.initial_state_params_list)
        return self._run_single(None)

    # ------------------------------------------------------------------
    # Internal helpers
    def _run_single(self, ic_params: Optional[Dict[str, Any]]) -> Tuple[_np.ndarray, Sequence[Dict[str, Any]]]:
        """Execute a single simulation run with optional initial condition parameters."""
        # initial state
        state0 = self.model.initial_state(ic_params)
        # apply BC before integration
        state0 = self.model.apply_boundary_conditions(state0)
        # convert to jax pytrees if jax and diffrax are available
        if _HAS_JAX and _HAS_DIFFRAX:
            def to_jax_tree(state):
                return {k: jnp.asarray(v) for k, v in state.items()}
            state0_jax = to_jax_tree(state0)
        else:
            state0_jax = state0
        # Define RHS function
        def rhs_fn(t, y, args):
            """Right‑hand side callback for Diffrax.

            The argument ``t`` may be a JAX tracer when JAX is used, in
            which case converting it to a Python ``float`` would cause
            a ``ConcretizationTypeError``.  We therefore pass ``t``
            directly to the model.  Our PDE models do not depend on
            ``t`` by default, so this is safe.  Boundary conditions
            are applied on the state before computing derivatives.
            """
            y_dict = {k: v for k, v in y.items()}
            y_dict = self.model.apply_boundary_conditions(y_dict)
            # Pass t directly; avoid float(t) for JAX compatibility
            dy_dict = self.model.rhs(y_dict, t)
            return dy_dict
        # choose solver (returns None if Diffrax/JAX is unavailable)
        solver = self._choose_solver()
        # set up saving for diffrax
        saveat = None
        if self.save_every is not None and _HAS_DIFFRAX:
            saveat = SaveAt(t1=True)
        # integrate using diffrax if available and solver is defined
        # Use Diffrax only if a solver is defined, JAX/Diffrax are available and
        # the discretiser backend is JAX.  Otherwise fall back to an explicit
        # Euler integrator.  This prevents attempting to use Diffrax with
        # NumPy arrays (which can lead to tracing errors).
        if solver is not None and _HAS_DIFFRAX and _HAS_JAX and getattr(self.model, 'diff', None) is not None and getattr(self.model.diff, 'backend', None) == 'jax':
            term = ODETerm(rhs_fn)  # type: ignore[misc]
            sol = diffeqsolve(term, solver, t0=self.t0, t1=self.t1,
                              dt0=self.dt0 if self.dt0 else None,
                              y0=state0_jax, saveat=saveat)
            saved_times = _np.array([self.t0, self.t1])
            # Convert JAX arrays back to NumPy for consistency
            saved_states = [state0, {k: _np.array(v) for k, v in sol.ys.items()}]
        else:
            # fallback explicit Euler integrator
            dt = self.dt0 if self.dt0 else 1e-3
            nsteps = int(_np.ceil((self.t1 - self.t0) / dt))
            t = self.t0
            state = state0
            saved_times: list[float] = []
            saved_states: list[Dict[str, Any]] = []
            for step in range(nsteps):
                if self.save_every is not None and step % self.save_every == 0:
                    saved_times.append(t)
                    saved_states.append({k: arr.copy() for k, arr in state.items()})
                dstate = self.model.rhs(state, t)
                # explicit Euler update
                for k in state:
                    state[k] = state[k] + dt * dstate[k]
                state = self.model.apply_boundary_conditions(state)
                t += dt
            # Always save final state
            saved_times.append(t)
            saved_states.append({k: arr.copy() for k, arr in state.items()})
        return _np.asarray(saved_times), saved_states

    def _run_multiple(self, ic_params_list: Sequence[Dict[str, Any]]):
        """Execute multiple simulation runs, one per set of initial parameters.

        When JAX and Diffrax are available and more than one initial
        condition is provided, this will attempt to vectorise the
        integration using ``jax.vmap``.  Otherwise, runs are executed
        sequentially.
        """
        # If JAX/diffrax available, vectorise solves
        n = len(ic_params_list)
        if n > 1 and _HAS_DIFFRAX and _HAS_JAX:
            # Build list of initial states (PyTrees)
            initial_states = [self.model.apply_boundary_conditions(self.model.initial_state(ic))
                              for ic in ic_params_list]
            # Convert to JAX PyTree and stack along new batch axis
            def to_jax_tree(state):
                return {k: jnp.asarray(v) for k, v in state.items()}
            jax_states = [to_jax_tree(s) for s in initial_states]
            # Stack each field into batch dimension
            batched_state = {}
            for key in jax_states[0].keys():
                batched_state[key] = jnp.stack([s[key] for s in jax_states], axis=0)
            # Build RHS that operates per instance; jax.vmap will broadcast over batch axis
            def rhs_fn(t, y, args):
                """Batched right‑hand side for multiple initial conditions.

                The argument ``t`` may be a JAX tracer when using JAX.
                We avoid converting it to a Python float to prevent
                ``ConcretizationTypeError``.  Boundary conditions are
                applied to each batch element using ``vmap`` and the
                model's ``rhs`` is called directly with the tracer ``t``.
                """
                # y is batched dict with leading dimension equal to batch size
                def apply_bcs_single(single_state: Dict[str, Any]) -> Dict[str, Any]:
                    return self.model.apply_boundary_conditions(single_state)
                bcs_applied = jax.vmap(apply_bcs_single)(y)
                def rhs_single(state_dict: Dict[str, Any]) -> Dict[str, Any]:
                    return self.model.rhs(state_dict, t)
                return jax.vmap(rhs_single)(bcs_applied)
            solver = self._choose_solver()
            term = ODETerm(rhs_fn)
            # Use vmap to solve each instance independently.
            def solve_single(y0):
                sol = diffeqsolve(term, solver, t0=self.t0, t1=self.t1,
                                   dt0=self.dt0 if self.dt0 else None,
                                   y0=y0, saveat=SaveAt(t1=True))
                return sol.ys
            # Use vmap to map solve_single over batch
            batched_sol = jax.vmap(solve_single)(batched_state)
            # Convert batched_sol to list of python dicts with numpy arrays
            results = []
            for i in range(n):
                final_state = {k: _np.array(v[i]) for k, v in batched_sol.items()}
                # Record initial time and final time; only final is saved
                times = _np.array([self.t0, self.t1])
                states = [initial_states[i], final_state]
                results.append((times, states))
            return results
        # Fallback: sequential runs
        results = []
        for ic in ic_params_list:
            results.append(self._run_single(ic))
        return results
