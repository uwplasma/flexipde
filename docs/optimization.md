# Optimisation and gradients

One of the unique features of FlexiPDE is that its solver is **differentiable**.
By using **JAX** for array operations and **Diffrax** for time integration,
the solution of a PDE can be treated as a function of its parameters or
initial conditions.  You can then compute derivatives of scalar objective
functions with respect to these parameters and use gradient–based
optimisation to fit models, discover equilibria, or train machine–learning
models.

## Computing gradients

The function `flexipde.optim.simulate_and_grad` runs a simulation and
computes the gradient of a user–defined objective with respect to the
simulation parameters.  For example, consider the one–dimensional advection
equation $\partial\_t u + c \partial\_x u = 0$.  Suppose you want to
compute how the mean value of $u$ at the final time depends on the initial
amplitude $A$ of a sinusoidal perturbation.  You can do so as follows:

```python
import jax.numpy as jnp
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import Advection
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad

grid = Grid.regular([(0.0, 2 * jnp.pi)], [64], periodic=[True])
diff = SpectralDifferentiator(grid, backend="jax")
model = Advection(grid, diff, velocity=[1.0])
sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.1)

def ic_from_param(p):
    return {"type": "sinusoidal", "amplitude": p, "wavevector": [1], "phase": 0.0, "backend": "jax"}

def objective_fn(final_state):
    return jnp.mean(final_state["u"])

param = jnp.array(1.0)
loss, grad = simulate_and_grad(sim, param, ic_from_param, objective_fn)
print(loss, grad)
```

The returned gradient tells you how a small change in the initial amplitude
affects the objective.  Internally, `simulate_and_grad` uses Diffrax’s
adjoint methods to differentiate through the numerical integrator.

## Parameter optimisation

To perform a full optimisation, use `flexipde.optim.optimize_params`.
This function wraps `simulate_and_grad` inside an iterative loop with an
Optax optimizer.  You supply a starting guess for the parameters and a
learning rate, and it returns a best–fit parameter along with the final
objective value.  For example, to optimise the thermal velocity in a
two–stream Vlasov simulation so that the growth rate of the instability is
minimal, see the `examples/optimise_vlasov_growth.py` script.

When using optimisation features you must install the `jax` extra and
ensure that both JAX and Diffrax are available.  If JAX is missing the
optimisation functions will raise an informative error message.