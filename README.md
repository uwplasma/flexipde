# flexipde

**flexipde** is a flexible, high‑performance solver framework for
plasma physics and other partial differential equations (PDEs).  It is
written in pure Python on top of
[JAX](https://github.com/google/jax),
[Equinox](https://github.com/patrick-kidger/equinox) and
[Diffrax](https://github.com/patrick-kidger/diffrax) and combines the
expressiveness of Python with the speed of just‑in‑time compilation and
automatic vectorisation.  The goal of the project is to make it as
easy as possible to specify and solve new PDEs while retaining
research‑grade performance on CPUs and GPUs.

The design is inspired by existing frameworks such as
BOUT++, which can evolve arbitrary numbers of fluid equations in
curvilinear geometry using runtime‑swappable numerical methods【501252340464299†L24-L33】, and
Φ‑Flow, a differentiable PDE solver that reuses the same code across
2D/3D and NumPy/PyTorch/TensorFlow/JAX backends【383486560010699†L364-L376】.  flexipde adopts
these ideas and builds on the modern JAX ecosystem: differential
equation solvers from Diffrax【805256974599970†L76-L91】, model abstractions from Equinox【549686066868328†L78-L97】,
and distributed array sharding and automatic parallelisation from JAX【21183366581013†L327-L435】.

## Getting started

flexipde can be used either through a declarative TOML configuration
file (via the command‑line interface) or directly from Python.  For
newcomers we provide several **Python driver scripts** in the
`examples/` directory.  These scripts illustrate how to set up
common PDEs, run simulations and visualise the results.  They are
intended to be pedagogical and can serve as starting points for
researchers wishing to build their own models.  Available examples
include:

* `run_advection.py` – one‑dimensional linear advection using a
  spectral scheme.
* `run_diffusion_1d.py` – diffusion on a line with Neumann boundary
  conditions using finite differences.
* `run_diffusion_2d.py` – diffusion on a square with Dirichlet
  boundaries and visualisation of the initial and final states.
* `run_diffusion_3d.py` – diffusion in a periodic cube with
  mid‑plane cross‑section plots.
* `run_alfven.py` – an ideal MHD Alfvén wave simulation with
  spectral derivatives.
* `run_cylindrical_diffusion.py` – a custom cylindrical diffusion
  model in $(r,z)$ coordinates, demonstrating how to implement
  axisymmetric diffusion with metric factors.
* `optimize_transport_equation.py` – optimisation of a variable‑
  coefficient transport equation using JAX and Optax to minimise
  time variation of the solution.

Each script can be run directly with Python (e.g.
`python examples/run_diffusion_2d.py`) and will print a summary of the
simulation as well as produce a plot using matplotlib.  Inspect these
scripts for examples of custom initial conditions, boundary condition
manipulation and model subclassing.

## Features

- **Modular design** – swap discretisation schemes (spectral, finite
  difference, finite element) at runtime without changing your model.
- **Curvilinear coordinates** – provide your own metric tensor; the
  grid stores metric information enabling simulations in arbitrary
  coordinate systems.
- **Arbitrary PDE systems** – subclass the base `PDEModel` and
  implement `initial_state()` and `rhs()` to define new equations.
  Alternatively use built‑in models for advection, diffusion and
  ideal MHD.
- **Rich initial conditions** – specify Gaussian, sinusoidal or
  constant profiles via configuration or supply custom callables.  When
  JAX is available, initial conditions can depend on JAX parameters
  enabling gradient‑based optimisation.
- **Differentiable and optimisable** – run simulations inside
  `jax.grad` or `jax.jit` to compute gradients of outputs with respect
  to input parameters.  The `flexipde.optim` module provides helper
  functions to compute gradients and to plug simulations into Optax
  optimisers.  This enables end‑to‑end optimisation and machine
  learning tasks such as physics‑informed neural networks, similar to
  the differentiable solvers in JAX‑Fluids【889881592370340†L35-L40】 and JAX‑MD【925522637824375†L27-L34】.
- **Configurable via TOML** – run simulations from a declarative
  configuration file or drive them programmatically from Python.
- **CPU/GPU support** – leverage JAX for transparent CPU/GPU
  execution; arrays can be sharded across multiple devices for high
  performance【21183366581013†L327-L435】.  Parallel runs over multiple initial
  conditions can be vectorised via `jax.vmap`.
- **Integration with Optax and Flax** – because the solver and models
  are differentiable and JIT‑compatible, they can be used in
  optimisation loops with [Optax](https://github.com/deepmind/optax) or
  inside neural networks built with [Flax](https://github.com/google/flax).
  For example, you can train a neural network to predict closure
  terms in a PDE and embed the solver in a larger model.
- **Open source** – MIT licensed and designed to be easily extended.

## Installation

flexipde is a pure‑Python package.  It requires Python 3.10 or later
and optionally JAX for accelerated computing.  You can install it
along with its dependencies via `pip`:

```bash
pip install jax jaxlib equinox diffrax sympy2jax jaxtyping tomli
pip install -e .  # install flexipde itself in editable mode
```

Without JAX the package will still run using NumPy and a simple
explicit Euler integrator, albeit more slowly.  We recommend
installing `jax[cpu]` or `jax[cuda]` as appropriate for your machine.

## Quick start

The easiest way to run a simulation is to write a TOML file that
describes your problem.  For example, the following file defines a
one‑dimensional linear advection of a sine wave on a periodic domain:

```toml
[grid]
domain = [[0.0, 6.283185307179586]]  # 0 to 2π
shape = [128]
periodic = [true]

[discretisation]
scheme = "spectral"
backend = "jax"

[model]
type = "advection"
velocity = [1.0]

[solver]
t0 = 0.0
t1 = 2.0
dt0 = 0.05
solver = "Dopri5"
save_every = 10
```

Run the simulation with

```bash
python -m flexipde.run path/to/advection.toml --output advection.npz
```

This will integrate the equation from `t=0` to `t=2` using a
fifth‑order Runge–Kutta method and save the solution at every
10th step into `advection.npz`.  You can inspect the results in
Python:

```python
import numpy as np
data = np.load('advection.npz')
times = data['times']
u = data['u']  # shape (n_times, n_points)
```

For more complex problems you can subclass `PDEModel` in Python and
provide custom `rhs` logic.  See `flexipde/models/ideal_mhd.py` or
`examples/run_cylindrical_diffusion.py` for an example of defining
your own physics (including coordinate‑dependent differential
operators).

### Optimisation and machine learning

Thanks to JAX's automatic differentiation and Diffrax's adjoint
methods【805256974599970†L81-L91】, flexipde can be used in optimisation
routines and machine learning loops.  The `flexipde.optim` module
provides a convenient function `simulate_and_grad` for computing the
gradient of a scalar objective with respect to parameters controlling
the initial condition or model.  This enables, for example, parameter
calibration, data assimilation and physics‑informed neural networks.

Below is a minimal example showing how to optimise the amplitude of a
sinusoidal initial condition in the advection equation to maximise the
mean value of the field at final time:

```python
import jax
import jax.numpy as jnp
import optax
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models.advection import LinearAdvection
from flexipde.solver import Simulation
from flexipde.optim import simulate_and_grad, optimize_params

# Build simulation (use JAX backend)
grid = Grid.regular([(0.0, 2 * jnp.pi)], [64], periodic=[True])
diff = SpectralDifferentiator(grid, backend="jax")
model = LinearAdvection(grid, diff, velocity=[1.0])
sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.1)

# Map parameter to initial condition dictionary
def ic_from_params(p):
    return {"type": "sinusoidal", "amplitude": p, "wavevector": [1], "phase": 0.0, "backend": "jax"}

# Objective: maximise mean value of u at final time
def objective_fn(final_state):
    u = final_state["u"]
    return jnp.mean(u)

# Optimise amplitude using stochastic gradient descent
init_amp = jnp.array(0.5)
opt = optax.sgd(learning_rate=0.1)
loss, opt_amp = optimize_params(sim, init_amp, ic_from_params, objective_fn, opt, num_steps=20)
print(float(loss), float(opt_amp))
```

This example follows JAX‑MD's functional design【925522637824375†L37-L40】: the
simulation state is represented as a PyTree and the optimisation
function returns both the loss and the updated parameters.  More
complex objectives can integrate data misfits, regularisation terms or
neural networks built with Flax.

## Documentation

Documentation is provided in the `docs/` folder and rendered using
[MkDocs](https://www.mkdocs.org/) with the Material theme and
MathJax support for typesetting equations.  The tutorials cover
installation, usage via TOML and Python, extending the library with
new models, and optimisation with Optax.  You can build the docs
locally via `mkdocs build` or view them online at the project’s
GitHub Pages site.

## Development and contributions

The project welcomes contributions.  To run the test suite, install
pytest and run

```bash
pytest tests
```

The continuous integration workflow defined in `.github/workflows/ci.yml`
runs the tests on push.

## License

flexipde is licensed under the MIT License.
