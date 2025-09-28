"""Optimise a variable‑coefficient 1D transport equation for stability.

This script illustrates how to couple flexipde with JAX and Optax to
perform gradient‑based optimisation of a PDE parameter.  We consider a
1D transport equation

.. math::

    \partial_t u + \partial_x\bigl(a(x,p)\, u\bigr) = 0,

where the advective velocity $a(x,p) = 1 + p\,\sin x$ depends on a
parameter ``p``.  We seek the value of ``p`` that minimises the
time variation of the solution, quantified here by the mean squared
difference between the final state and the initial state.  The
gradient of the loss with respect to ``p`` is computed via automatic
differentiation through the time integration using diffrax.  The
parameter is updated using Optax gradient descent.

Run this script with::

    python examples/optimize_transport_equation.py

Note: this example requires optional dependencies JAX, diffrax and
optax.  If these are not installed, the script will notify the user
and exit gracefully.

"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Sequence

try:
    import jax
    import jax.numpy as jnp
    from jax import value_and_grad
    import optax
except Exception as exc:  # pragma: no cover - runtime dependency check
    jax = None  # type: ignore
    jnp = None  # type: ignore
    optax = None  # type: ignore

from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.base import PDEModel, FieldBC
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad


class TransportEquation(PDEModel):
    """1D transport equation with a sinusoidal variable coefficient."""

    param: Any = 0.0  # The optimisation parameter controlling a(x,p)
    init_u: Any | None = None

    def __post_init__(self) -> None:
        # Periodic boundary condition for u
        self.field_bcs = {"u": FieldBC("periodic")}

    def initial_state(self, ic_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        # If a custom function was supplied use it
        if self.init_u is not None:
            coords = self.grid.coordinate_arrays("jax" if isinstance(self.param, jax.Array) else "numpy")
            u0 = self.init_u(coords)
            return {"u": u0}
        # Otherwise default to a sine wave along x
        coords = self.grid.coordinate_arrays("jax" if isinstance(self.param, jax.Array) else "numpy")
        x = coords[0]
        np_module = jnp if isinstance(self.param, jax.Array) else np
        u0 = np_module.sin(x)
        return {"u": u0}

    def rhs(self, state: Dict[str, Any], t: Any) -> Dict[str, Any]:
        # Compute the PDE RHS: ∂t u = -∂x(a(x,p) u)
        u = state["u"]
        # Choose backend based on type of param
        use_jax = isinstance(self.param, jax.Array) if jax is not None else False
        coords = self.grid.coordinate_arrays("jax" if use_jax else "numpy")
        x = coords[0]
        if use_jax:
            a = 1.0 + self.param * jnp.sin(x)
            flux = a * u
            dflux_dx = self.diff.grad(flux, 0)
            return {"u": -dflux_dx}
        else:
            a_np = 1.0 + float(self.param) * np.sin(x)
            flux_np = a_np * u
            dflux_dx_np = self.diff.grad(flux_np, 0)
            return {"u": -dflux_dx_np}


def main() -> None:
    # Check optional dependencies
    if jax is None or optax is None:
        print(
            "JAX and Optax are required for this optimisation example. "
            "Please install them to run this script."
        )
        return
    # 1D periodic grid on [0,2π)
    grid = Grid.regular([(0.0, 2 * np.pi)], [128], periodic=[True])
    # Use a spectral discretiser for differentiability
    diff = SpectralDifferentiator(grid, backend="jax")
    # Instantiate the PDE model.  We will set the optimisation parameter
    # afterwards to avoid passing it into the dataclass constructor (which
    # only accepts grid, diff and field_bcs from its base class).
    param = jnp.array(0.5)
    model = TransportEquation(grid, diff)
    model.param = param
    # Set up the simulation
    sim = Simulation(model, t0=0.0, t1=2.0, dt0=0.05)

    # Define the initial condition generator from the parameter.  In this
    # case the parameter does not affect the initial condition so we
    # ignore it.
    def ic_from_param(p: Any) -> Dict[str, Any]:
        # Always use a sine wave as initial state
        return {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1], "phase": 0.0, "backend": "jax"}

    # Define the objective function: minimise the change between final and initial state
    def objective_fn(final_state: Dict[str, Any], initial_state: Dict[str, Any]) -> Any:
        u_final = final_state["u"]
        u_initial = initial_state["u"]
        # Mean squared difference between final and initial state
        return jnp.mean((u_final - u_initial) ** 2)

    # Set up optimiser
    opt = optax.adam(learning_rate=0.1)
    opt_state = opt.init(param)
    max_iters = 20
    for i in range(max_iters):
        # Create a closure to compute loss and gradient
        def loss_fn(p: Any) -> Any:
            # Update model parameter
            model.param = p
            # Generate initial state
            ic_params = ic_from_param(p)
            # Simulate and compute gradient using simulate_and_grad
            # simulate_and_grad expects a scalar objective from final_state
            def obj_with_init(final_state: Dict[str, Any]) -> Any:
                # Convert initial state for this parameter
                initial_state = model.initial_state(ic_params)
                return objective_fn(final_state, initial_state)
            loss, grad = simulate_and_grad(sim, p, ic_from_param, obj_with_init)
            return loss, grad
        # Compute loss and gradient
        loss, grad = loss_fn(param)
        # Update parameter using Optax
        updates, opt_state = opt.update(grad, opt_state, param)
        param = optax.apply_updates(param, updates)
        print(f"Iter {i+1:02d}: loss = {float(loss):.6f}, param = {float(param):.4f}")
    # Report optimised parameter
    print(f"Optimisation complete: p ≈ {float(param):.4f}")


if __name__ == "__main__":
    main()
