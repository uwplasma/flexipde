# Optimisation and automatic differentiation

One of the unique strengths of `flexipde` is its ability to compute
gradients of simulation outputs with respect to inputs.  Thanks to
JAX and Diffrax, the solver behaves as a pure function of its
parameters; the entire integration is differentiable using
automatic differentiation【805256974599970†L81-L91】.  This enables a wide range of
applications, from data‑driven model calibration to physics‑informed
neural networks.

## Computing gradients of a simulation

The function `simulate_and_grad` in `flexipde.optim` runs a single
simulation and returns both the objective and its gradient with
respect to user‑supplied parameters.  The parameters can be any
PyTree (e.g. scalar, array or dictionary) that `ic_from_params`
maps to a dictionary of initial condition parameters.  The
`objective_fn` computes a scalar from the final state (for example, a
norm or loss with respect to a target).

```python
from flexipde.optim import simulate_and_grad
from flexipde.solver import Simulation
from flexipde.models import Advection
from flexipde.discretisation import SpectralDifferentiator
from flexipde import Grid

# Build a simulation (JAX backend required)
grid = Grid.regular([(0.0, 2*3.141592653589793)], [64], periodic=[True])
diff = SpectralDifferentiator(grid, backend="jax")
model = Advection(grid, diff, velocity=[1.0])
sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.01, solver="Dopri5")

# Define a parameterised initial condition and objective
def ic_from_params(p):
    return {"amplitude": p}

def objective_fn(final_state):
    u_final = final_state["u"]
    return jnp.sum((u_final - 0.0)**2)  # minimise final amplitude

# Run the simulation and compute gradient
loss, grad = simulate_and_grad(sim, params=0.1, ic_from_params=ic_from_params,
                               objective_fn=objective_fn)
print("loss:", loss)
print("gradient:", grad)
```

When using JAX, `simulate_and_grad` employs Diffrax’s
``BacksolveAdjoint`` to compute gradients efficiently without
differentiating through each solver step【805256974599970†L81-L91】.

## Optimising parameters

You can perform iterative optimisation by combining
`simulate_and_grad` with an Optax optimiser.  The function
`optimize_params` in `flexipde.optim` implements a simple loop that
updates parameters for a fixed number of steps.  For example,
suppose you wish to find the thermal velocity that maximises the
growth of the two–stream instability.  You could define the loss as
the negative of the electric field energy at the end of the
simulation and run gradient ascent:

```python
import optax
from flexipde.optim import optimize_params
from flexipde.models import VlasovTwoStream

# Build the Vlasov simulation as in the examples
grid = Grid.regular([(0.0, 2*3.141592653589793)], [128], periodic=[True])
diff = SpectralDifferentiator(grid, backend="jax")
model = VlasovTwoStream(grid, diff, nv=64, v_min=-5.0, v_max=5.0)
sim = Simulation(model, t0=0.0, t1=10.0, dt0=0.1, solver="Dopri5")

# Parameter: thermal velocity ratio
init_p = 1.0

def ic_from_p(p):
    return {"thermal_velocity": p, "amplitude": 1e-3}

def objective(final_state):
    # Compute electric field energy from distribution f
    f = final_state["f"]
    # Reconstruct electric field via Poisson solver
    E = model._poisson_field(f, jnp)
    return -jnp.mean(E**2)  # maximise growth => minimise negative energy

# Optimiser (gradient ascent)
opt = optax.adam(learning_rate=0.1)
loss, p_opt = optimize_params(sim, init_params=init_p,
                              ic_from_params=ic_from_p,
                              objective_fn=objective,
                              optimizer=opt,
                              num_steps=20)
print("Optimal thermal_velocity:", p_opt)
```

This example illustrates how to carry out parameter optimisation on a
full PDE simulation.  More advanced strategies, such as scheduling
learning rates or imposing constraints, can be implemented by
customising the optimisation loop.

## Beyond gradients

The ability to differentiate through entire simulations opens the
door to machine‑learning applications.  For instance, you can
integrate neural networks via [Flax](https://flax.readthedocs.io/)
into your models, or embed `flexipde` simulations inside larger
physics‑informed neural networks.  JAX‑MD and JAX‑Fluids provide
inspiration for differentiable model design【925522637824375†L27-L40】【889881592370340†L35-L41】.
We encourage contributions that extend these ideas to plasma physics.