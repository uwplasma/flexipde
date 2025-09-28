# User guide

This guide walks you through building and running simulations with
`flexipde`, from constructing a grid and choosing a discretisation to
configuring models and initial conditions.  You can run simulations
either from Python code or from declarative TOML configuration files.

## Creating a grid

All simulations start with a spatial grid.  A `Grid` stores the
coordinates of each grid point and, optionally, the metric tensor for
curvilinear coordinates.  To create a regular Cartesian grid use the
`Grid.regular` factory:

```python
from flexipde import Grid

# Create a 1D grid on [0, 2π) with 128 points and periodic boundaries
grid = Grid.regular(domain=[(0.0, 2 * 3.141592653589793)],
                    shape=[128],
                    periodic=[True])

# For 2D or 3D domains, provide a domain and shape per dimension
grid2 = Grid.regular(domain=[(0.0, 1.0), (0.0, 1.0)],
                     shape=[64, 64],
                     periodic=[True, True])
```

You can also load a grid from a configuration dictionary or TOML file
using `Grid.from_config`.  See [`flexipde/grid.py`](../flexipde/grid.py)
for details.

## Choosing a discretisation

`flexipde` currently supports two discretisation schemes:

* **Spectral** – uses the Fast Fourier Transform to compute
  derivatives.  It is highly accurate for smooth functions and
  periodic domains.  Based on the wavenumber array computed from the
  grid spacing and shape.
* **Finite difference** – uses centred finite difference stencils
  (second order) to compute derivatives.  Supports periodic and
  non‑periodic boundary conditions.

To instantiate a discretiser, pass the grid and optionally a
backend:

```python
from flexipde.discretisation import SpectralDifferentiator, FiniteDifference

diff_spec = SpectralDifferentiator(grid, backend="jax")  # FFTs in JAX
diff_fd = FiniteDifference(grid)                          # second order FD
```

The `backend` argument controls whether derivatives are computed with
NumPy or JAX.  If omitted, the discretiser chooses NumPy by default.

## Selecting a model

Models encapsulate the physics of your problem.  Each model
subclasses `PDEModel` and implements an `initial_state` method that
returns a dictionary of field arrays and an `rhs` method that
computes the time derivative of those fields.  The following models
are included:

| Model            | Description                                                                  |
|------------------|------------------------------------------------------------------------------|
| `Advection`      | Linear advection of a scalar field by a constant velocity vector.            |
| `Diffusion`      | Scalar diffusion equation with constant diffusivity.                        |
| `IdealAlfven`    | Simplified ideal MHD: transverse Alfvén waves propagating along a background field. |
| `VlasovTwoStream`| 1D Vlasov–Poisson solver for the two–stream instability.                     |

To create a model, pass the grid, discretiser and model‑specific
parameters.  For example, a 1D advection model with velocity 1.0:

```python
from flexipde.models import Advection

model = Advection(grid, diff_spec, velocity=[1.0])
```

Refer to [Models](models.md) for a detailed description of each model
and its parameters.

### Initial conditions

Each model supports custom initial conditions via its
`initial_state(ic_params)` method.  The meaning of `ic_params`
depends on the model.  For example, the advection model accepts
``type`` (``"gaussian"``, ``"sinusoidal"`` or ``"constant"``), ``amplitude``,
and other keys.  See the model documentation for details.

When using the configuration file interface, you can specify an
``[initial_conditions]`` section with parameters that will be passed
directly to `initial_state`.  You can also provide a list of initial
condition tables to run multiple simulations in one go.

## Running a simulation from Python

The `Simulation` class orchestrates time integration.  You must
specify the start and end time, an optional initial step size, the
solver and the model.  For example:

```python
from flexipde.solver import Simulation

sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.01, solver="Dopri5")
times, states = sim.run()
```

`times` is an array of save times and `states` is a list of
dictionaries holding the fields at those times.  When JAX and
Diffrax are available, Diffrax’s adaptive solvers such as
`Dopri5` and `Tsit5` are used【805256974599970†L81-L91】.  Otherwise a simple
explicit Euler integrator is used as a fallback.

You can supply a list of initial condition parameters to run a batch
of simulations.  If JAX and Diffrax are available, the runs are
vectorised with `jax.vmap` for parallel execution.

## Running from a configuration file

For reproducibility and ease of use, `flexipde` accepts TOML files
that describe the grid, discretisation, model, solver and initial
conditions.  See the [examples](examples.md) for complete files.  To
run a simulation from the command line:

```bash
python -m flexipde.run path/to/config.toml --output results.pkl.gz
```

The script will read the configuration, build the simulation, run it
and save the results.  If `rich` is installed, summary information is
printed in a colourful table.  Without an output file, the script
prints a summary of the saved times and field extrema.

Saved results are encapsulated in a `SimulationResult` object and can
be reloaded for analysis and plotting.