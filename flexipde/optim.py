"""
Optimization utilities for flexipde.

This module provides helper functions for computing gradients of
simulation outputs with respect to input parameters and for running
optimization loops using Optax.  It leverages JAX's automatic
differentiation together with Diffrax's adjoint methods to enable
efficient backpropagation through time integrators.

Background
----------
JAX‑Fluids demonstrates that fully differentiable solvers enable
end‑to‑end optimization of PDE models【889881592370340†L35-L40】.  JAX‑MD emphasises
functional, data‑driven design where data are represented as arrays and
transformed by pure functions【925522637824375†L37-L40】.  These principles guide
the functions below: the simulation is treated as a pure function of
its inputs and can therefore be differentiated with respect to those
inputs using ``jax.grad``.  Diffrax provides adjoint methods such as
``BacksolveAdjoint`` to compute gradients efficiently through ODE
solvers【805256974599970†L81-L91】.

The main entry point is :func:`simulate_and_grad`, which runs a
``Simulation`` for a single set of initial parameters, computes a
user‑defined scalar objective from the final state, and returns the
objective and its gradient with respect to the parameters.

Users can build more complex optimization routines by combining
``simulate_and_grad`` with Optax optimizers.  For example, to
calibrate the amplitude of an initial sine wave in an advection model
one might minimise the squared difference between the simulated final
state and some target profile.  See the ``examples/`` directory for
illustrations.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Tuple

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
    from diffrax import ODETerm, BacksolveAdjoint, diffeqsolve
except Exception:
    ODETerm = None  # type: ignore
    BacksolveAdjoint = None  # type: ignore
    diffeqsolve = None  # type: ignore

from flexipde.solver import Simulation

def simulate_and_grad(
    sim: Simulation,
    params: Any,
    ic_from_params: Callable[[Any], Dict[str, Any]],
    objective_fn: Callable[[Dict[str, Any]], jnp.ndarray],
) -> Tuple[jnp.ndarray, Any]:
    """Run a simulation and compute the gradient of a scalar objective.

    Parameters
    ----------
    sim : :class:`Simulation`
        The simulation object configured with a model, grid and
        solver.  Only the fields ``t0``, ``t1``, ``dt0`` and
        ``solver`` are used.  The simulation must be compatible with
        JAX and Diffrax; that is, its model should generate JAX
        arrays and the discretisation should support JAX.
    params : PyTree
        Input parameters controlling the initial condition.  ``params``
        may be any JAX‑compatible container (e.g. scalar, array,
        dictionary) that ``ic_from_params`` maps to an ``ic_params``
        dictionary understood by ``sim.model.initial_state``.
    ic_from_params : callable
        Function mapping ``params`` to a dictionary of initial
        condition parameters.  This allows flexible parameterisations
        of the initial state.  The returned dictionary should be
        compatible with the model's ``initial_state`` method and
        should request ``backend='jax'`` if necessary.
    objective_fn : callable
        Function taking the final state dictionary (mapping field names
        to JAX arrays) and returning a scalar ``jnp.ndarray``.  This
        defines the objective whose gradient with respect to ``params``
        will be computed.

    Returns
    -------
    (loss, grad)
        A tuple containing the scalar objective value and its gradient
        with respect to ``params``.

    Notes
    -----
    This function uses Diffrax's ``BacksolveAdjoint`` method by
    default.  This approach stores the integration trajectory and
    solves a backwards adjoint ODE to compute gradients without
    differentiating through every solver step.  See the Diffrax
    documentation for details【805256974599970†L81-L91】.
    """
    if not _HAS_JAX or diffeqsolve is None or ODETerm is None:
        raise RuntimeError(
            "JAX and Diffrax must be installed to compute gradients."
        )
    # Choose solver from simulation
    solver = sim._choose_solver()
    # Build a loss function of params that returns objective
    def loss_fn(p):
        """Construct the loss for a given parameter PyTree.

        This inner function builds the initial state from the parameters,
        runs the simulation and returns the objective.  It avoids
        converting the JAX time tracer to a Python float, which would
        otherwise trigger a concretisation error.  It also ensures a
        ``SaveAt`` is provided to Diffrax so that the integration can
        determine the collection times in a JIT‑friendly manner.
        """
        ic_params = ic_from_params(p)
        # Ensure backend is jax for differentiability.  We copy the
        # dictionary so that we do not mutate the caller's object.
        if isinstance(ic_params, dict):
            ic_params = dict(ic_params)
            ic_params.setdefault("backend", "jax")
        # Create initial state using the model (JAX arrays)
        state0 = sim.model.initial_state(ic_params)
        # Apply boundary conditions once to the initial state
        state0 = sim.model.apply_boundary_conditions(state0)
        # Convert to JAX PyTree of arrays
        y0 = {k: jnp.asarray(v) for k, v in state0.items()}
        # Define RHS using model.rhs without converting the time argument
        def rhs_fn(t, y, args):
            # Apply boundary conditions on the fly
            y_dict = {kk: vv for kk, vv in y.items()}
            y_dict = sim.model.apply_boundary_conditions(y_dict)
            dy = sim.model.rhs(y_dict, t)
            return dy
        term = ODETerm(rhs_fn)
        # Provide a SaveAt object to record only the final time.  Passing
        # None triggers an AttributeError in diffrax when tracing.
        from diffrax import SaveAt  # type: ignore
        sol = diffeqsolve(
            term,
            solver,
            t0=sim.t0,
            t1=sim.t1,
            dt0=sim.dt0 if sim.dt0 else None,
            y0=y0,
            adjoint=BacksolveAdjoint(),
            saveat=SaveAt(t1=True),
        )
        # Extract the final state.  ``sol.ys`` is a PyTree matching
        # ``y0``; here we assume it is a dict mapping field names to
        # JAX arrays.
        final_state = {k: sol.ys[k] for k in sol.ys.keys()}
        return objective_fn(final_state)
    # Compute value and gradient
    loss, grad = jax.value_and_grad(loss_fn)(params)
    return loss, grad

def optimize_params(
    sim: Simulation,
    init_params: Any,
    ic_from_params: Callable[[Any], Dict[str, Any]],
    objective_fn: Callable[[Dict[str, Any]], jnp.ndarray],
    optimizer,
    num_steps: int,
) -> Tuple[jnp.ndarray, Any]:
    """Run a simple gradient‑based optimisation loop using Optax.

    Parameters
    ----------
    sim : Simulation
        The simulation to run (must be JAX‑compatible).
    init_params : PyTree
        Initial guess for the parameters to optimise.
    ic_from_params : callable
        Function mapping parameters to initial condition dictionaries.
    objective_fn : callable
        Function mapping the final state to a scalar loss.
    optimizer : optax.GradientTransformation
        An Optax optimizer (e.g. ``optax.adam`` or ``optax.sgd``).
    num_steps : int
        Number of optimization iterations.

    Returns
    -------
    (loss, params)
        The final loss and the optimised parameters.

    Notes
    -----
    The optimisation loop is written in a functional style, similar
    to the update loops in JAX‑MD【925522637824375†L220-L230】.  The state of the
    optimiser is carried through each iteration.  Users may wish to
    adapt this loop to their specific needs (e.g. by adding
    convergence checks or logging).
    """
    if not _HAS_JAX:
        raise RuntimeError("JAX is required for optimisation routines")
    import optax  # deferred import to avoid dependency if unused
    params = init_params
    opt_state = optimizer.init(params)
    for step in range(num_steps):
        loss, grads = simulate_and_grad(sim, params, ic_from_params, objective_fn)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
    final_loss, _ = simulate_and_grad(sim, params, ic_from_params, objective_fn)
    return final_loss, params