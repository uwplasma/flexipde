---
title: Optimisation and differentiable simulations
---

# Optimisation and differentiable simulations

One of the most powerful features of flexipde is its seamless integration with the JAX ecosystem.  When installed with the optional `jax` extras, all numerical kernels are compiled just‑in‑time for CPU or GPU execution and, crucially, they are differentiable.  This enables gradient‑based optimisation of simulation parameters and data‑driven discovery of governing equations.

## Computing gradients

The :mod:`flexipde.optim` module exposes a function `simulate_and_grad(sim, params, ic_from_params, objective_fn)` which takes a :class:`flexipde.solver.Simulation` object, a set of differentiable parameters, a function converting parameters into initial conditions, and a scalar objective function of the final state.  It returns both the objective value and the gradient with respect to the parameters.  Under the hood this uses Diffrax’s adjoint method and JAX automatic differentiation.

Example: optimise the amplitude of an initial sinusoidal perturbation so that the mean of the final field is zero:

```python
import jax
import jax.numpy as jnp
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import LinearAdvection
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad

grid = Grid.regular([(0.0, 2*jnp.pi)], [64], periodic=[True])
diff = SpectralDifferentiator(grid, backend="jax")
model = LinearAdvection(grid, diff, velocity=[1.0])
sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.1)

def ic_from_params(p):
    return {"type": "sinusoidal", "amplitude": p, "wavevector": [1], "phase": 0.0, "backend": "jax"}

def objective_fn(final_state):
    return jnp.mean(final_state["u"])**2

amp = jnp.array(1.0)
loss, grad = simulate_and_grad(sim, amp, ic_from_params, objective_fn)
print("loss", loss, "gradient", grad)
```

The returned gradient can be used with an optimiser such as Optax to perform gradient descent or more advanced algorithms.

## Automated optimisation

For convenience, the `optimize_params` function wraps `simulate_and_grad` and runs an optimisation loop with a specified number of steps and learning rate.  See `examples/optimise_vlasov_growth.py` for optimising the temperature ratio in a two‑stream Vlasov simulation and `examples/optimize_transport_equation.py` for finding the diffusion coefficient that minimises the time variation in a transport equation.

## Limitations

- Gradients are only available when using the spectral differentiator with `backend="jax"`.  The finite difference backend falls back to a pure NumPy implementation and is not differentiable.
- Some stiff systems may benefit from implicit or semi‑implicit integrators; Diffrax provides such solvers, but their differentiation support may vary.