# User guide

This guide introduces the key concepts of **flexipde**: grids and coordinates,
discretisation schemes, models and running simulations.  Newcomers to
computational physics can follow along to build their first simulation.

## Grids and coordinates

A simulation domain is discretised into a structured grid.  Create a
uniform grid using :class:`flexipde.grid.Grid.regular`, specifying the domain
intervals, number of points and periodicity for each dimension::

    from flexipde.grid import Grid
    grid = Grid.regular([(0.0, 2.0 * np.pi)], [64], [True])

This example constructs a one‑dimensional periodic grid on ``[0, 2π]`` with
64 points.  The coordinates are stored in ``grid.coords`` and the spacing
is returned by ``grid.spacing()``.

## Discretisation schemes

Two derivative operators are provided:

- **Spectral**: Uses the Fourier transform to compute derivatives exactly
  on periodic domains.  Choose this when high accuracy is required and
  periodic boundaries are appropriate.
- **Finite difference**: Uses central difference stencils to approximate
  derivatives on structured grids.  One‑sided stencils enforce Dirichlet
  or Neumann conditions at boundaries.

Create a discretiser as follows::

    from flexipde.discretisation import SpectralDifferentiator, FiniteDifference
    diff = SpectralDifferentiator(grid, backend="numpy")

The ``backend`` can be ``numpy`` or ``jax``.  When using JAX the
derivative operations are JIT compiled and run on GPU/TPU.

## Models

Models define the right‑hand side of a PDE and must inherit from
:class:`flexipde.models.base.PDEModel`.  The built‑in models include:

* **LinearAdvection**: Transport of a scalar field by a constant velocity.
* **Diffusion**: Heat equation with a constant diffusivity.
* **ResistiveMHD**: Simplified 1D resistive magnetohydrodynamics.
* **TwoFluid**: Separate advection of ion and electron densities.
* **DriftKinetic**: Simplified drift–kinetic equation in phase space.
* **IdealAlfven**: Propagation of Alfvén waves.
* **VlasovTwoStream**: 1D Vlasov–Poisson two‑stream instability.

You can subclass :class:`PDEModel` to implement your own model.  See the
examples for custom models such as Burgers' equation and the
Hasegawa–Wakatani system.

## Running simulations

To run a simulation, either write a TOML configuration file or build the
components directly in Python.

### From a configuration file

```
[grid]
domain = [[0.0, 2.0]]
shape = [64]
periodic = [true]

[discretisation]
type = "spectral"
backend = "numpy"

[model]
type = "diffusion"
[model.parameters]
diffusivity = 0.1

[simulation]
t0 = 0.0
t1 = 1.0
dt0 = 0.01

[initial_conditions]
u = { type = "sinusoidal", amplitude = 1.0, wavevector = [1], phase = 0.0 }
```

Save this as ``mydiffusion.toml`` and run::

    flexipde mydiffusion

### From Python

```python
from flexipde.grid import Grid
from flexipde.discretisation import SpectralDifferentiator
from flexipde.models import Diffusion
from flexipde.solver import Simulation

grid = Grid.regular([(0.0, 2.0)], [64], [True])
diff = SpectralDifferentiator(grid)
model = Diffusion(grid, diff, diffusivity=0.1)
sim = Simulation(model, t0=0.0, t1=1.0, dt0=0.01)
sim.initial_state_params = {"u": {"type": "sinusoidal", "amplitude": 1.0, "wavevector": [1]}}
result = sim.run()
```

See the examples and API reference for more usage.