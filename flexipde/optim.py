"""Optimisation helpers for flexipde.

This module provides functions to compute gradients of a scalar objective
with respect to simulation parameters and perform simple gradient descent
optimisation.  These helpers require JAX and Diffrax to be installed.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

try:
    import jax
    import jax.numpy as jnp
    from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt  # type: ignore[import]
    import optax  # type: ignore[import]
    from jaxtyping import PyTree  # type: ignore[import]
    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False

from .solver import Simulation
from .models.base import PDEModel


def simulate_and_grad(sim: Simulation,
                      params: Any,
                      ic_from_params: Callable[[Any], Dict[str, Any]],
                      objective_fn: Callable[[Dict[str, Any]], float]) -> Tuple[float, Any]:
    """Compute the loss and gradient with respect to parameters.

    This function assumes that JAX and Diffrax are available and that
    ``sim`` uses a JAX backend discretiser.  It builds a differentiable
    simulation under the hood and computes the gradient of ``objective_fn``
    applied to the final state with respect to ``params``.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("simulate_and_grad requires JAX, Diffrax and Optax")
    # closure capturing sim, ic_from_params, objective_fn
    def loss_fn(p):
        # set initial state based on parameters p (a JAX array)
        ic = ic_from_params(p)
        # ensure sim uses this initial state
        sim.initial_state_params = ic
        # run simulation (single)
        result = sim._run_single()
        # compute scalar objective from final state
        final_state = result.states[-1]
        return objective_fn(final_state)
    # compute value and gradient
    value, grad = jax.value_and_grad(loss_fn)(params)
    return float(value), grad


def optimize_params(sim: Simulation,
                    params0: Any,
                    ic_from_params: Callable[[Any], Dict[str, Any]],
                    objective_fn: Callable[[Dict[str, Any]], float],
                    lr: float = 0.1,
                    max_iters: int = 50) -> Tuple[float, Any]:
    """Perform simple gradient descent optimisation on simulation parameters.

    Parameters
    ----------
    sim:
        The simulation to run.
    params0:
        Initial parameter value (JAX array or Python float).
    ic_from_params:
        Function that maps parameter value to initial conditions dictionary.
    objective_fn:
        Function mapping the final state to a scalar loss.
    lr:
        Learning rate for gradient descent.
    max_iters:
        Number of gradient descent steps.

    Returns
    -------
    tuple
        A pair ``(loss, params)`` giving the final loss and the optimised
        parameters.
    """
    if not _JAX_AVAILABLE:
        raise RuntimeError("optimize_params requires JAX, Diffrax and Optax")
    # create optimizer
    optim = optax.adam(lr)
    opt_state = optim.init(params0)
    params = params0
    for _ in range(max_iters):
        loss, grad = simulate_and_grad(sim, params, ic_from_params, objective_fn)
        updates, opt_state = optim.update(grad, opt_state)
        params = optax.apply_updates(params, updates)
    loss, _ = simulate_and_grad(sim, params, ic_from_params, objective_fn)
    return loss, params