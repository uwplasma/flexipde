# Installation

FlexiPDE can be installed from PyPI or from source.  The core package only
depends on **NumPy** and (for Python versions older than 3.11) **tomllib**.
All advanced functionality—JAX acceleration, differentiable solvers, symbolic
equations, testing, and documentation—lives in optional extras.  To install
FlexiPDE with only the core dependencies:

```bash
pip install flexipde
```

To install with JAX and the optional physics and optimisation tools, use the
`jax` extra:

```bash
pip install 'flexipde[jax]'
```

The `symbolic` extra adds support for parsing equations from strings via
SymPy (through the `sympy2jax` library), and the `testing` extra installs
`pytest` and `mypy` for running the test suite and static type checking.

To install from a local checkout of the repository (for development), run

```bash
git clone https://github.com/uwplasma/flexipde.git
cd flexipde
pip install -e .
```

If you intend to build the documentation locally, install the `docs` extra and
run `mkdocs serve` from the project root:

```bash
pip install 'flexipde[docs]'
mkdocs serve
```

This will spin up a local development server at `http://localhost:8000` where
you can browse the documentation, which includes many examples and a thorough
API reference.