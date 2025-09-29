# flexipde

[![CI](https://github.com/uwplasma/flexipde/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/flexipde/actions/workflows/ci.yml)
[![Docs](https://github.com/uwplasma/flexipde/actions/workflows/docs.yml/badge.svg)](https://uwplasma.github.io/flexipde)
[![PyPI version](https://badge.fury.io/py/flexipde.svg)](https://pypi.org/project/flexipde)
[![codecov](https://codecov.io/gh/uwplasma/flexipde/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/flexipde)

**flexipde** is a flexible, high‑performance library for solving plasma physics equations in any number of dimensions.  It combines the ease of writing Python with the speed of modern array libraries like NumPy and JAX.  The library draws inspiration from—and goes beyond—established simulation frameworks such as **BOUT++** for fluid plasmas and **Gkeyll** for kinetic models【635715314594784†L14-L33】.  Instead of low‑level C++ or Fortran, you write concise Python code that can automatically run on CPUs or GPUs via JAX, and you can differentiate simulations with respect to parameters to enable optimisation or machine‑learning pipelines.

## Features

- **Unified grid and coordinate system:** Create structured grids in 1D, 2D or 3D, including cylindrical coordinate systems with arbitrary metrics.
- **Multiple discretisation methods:** Choose between Fourier spectral methods for periodic domains and finite difference methods for general boundaries.  Additional schemes (e.g. finite volume/WENO) can be added easily.
    - **Built‑in models:** Includes linear advection, diffusion, ideal and resistive MHD, two‑fluid plasmas, drift–kinetic equations and a Vlasov–Poisson solver for the two–stream instability.  Additional models can be implemented by subclassing a base class.
- **Custom equations:** Write your own PDEs without modifying the library—see the Burgers and Hasegawa–Wakatani examples.
- **JAX acceleration and differentiability:** When JAX is installed, simulations are JIT‑compiled and run on GPU/TPU.  You can compute gradients of objectives with respect to simulation parameters and plug them into optimisation routines (via Optax).
- **CLI and configuration files:** Run simulations from the command line using TOML configuration files.  You can specify the filename without the `.toml` suffix for convenience.
- **Extensive documentation:** A pedagogical guide introduces the mathematics and numerics behind each model, with worked examples and a gentle introduction for students and researchers.
- **Continuous integration and coverage:** GitHub workflows run tests, build the docs, and report coverage to codecov.

## Installation

```bash
pip install flexipde
# Or install with JAX support
pip install 'flexipde[jax]'
```

To install from source for development:

```bash
git clone https://github.com/uwplasma/flexipde.git
cd flexipde
pip install -e .[jax,testing,docs]
```

## Quick start

Run a 1D diffusion simulation from a TOML file:

```bash
flexipde examples/diffusion_1d
# automatically appends `.toml` if missing
```

Or run the same simulation from Python:

```python
from flexipde.io import build_simulation

sim = build_simulation('examples/diffusion_1d.toml')
times, states = sim.run()

import matplotlib.pyplot as plt
plt.plot(sim.grid.coords[0], states[-1]['u'])
plt.show()
```

See the `examples` directory for more scripts illustrating custom models and optimisation tasks.

## Documentation

The full documentation, including a tutorial introduction, API reference, and mathematical derivations, is available at **https://uwplasma.github.io/flexipde**.  The docs are built with MkDocs and use MathJax to render equations.

## Tests and coverage

To run the test suite with coverage:

```bash
pip install 'flexipde[testing]'
pytest --cov=flexipde tests
```

Manufactured solution tests are provided for each model to verify that spatial operators and PDE right‑hand sides are correct.  We use functions like `f(x,t)=cos(2x)*cos(3t)` to ensure the gradients, divergences and Laplacians match analytic expressions.

## Contributing

Contributions are welcome!  Please open issues or pull requests on GitHub.  We follow PEP8 coding style and maintain a high level of test coverage.

## License

This project is licensed under the MIT license—see the [LICENSE](LICENSE) file for details.