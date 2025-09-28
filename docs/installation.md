# Installation

`flexipde` can be installed from the Python Package Index or from the
source repository.  Depending on your needs you may also want to
install optional extras for JAX, optimisation or documentation.

## Prerequisites

* Python ≥ 3.10
* A C compiler if you plan to build dependencies from source

## Installing from PyPI

The simplest way to install the library is via `pip`:

```bash
pip install flexipde
```

This will install the core functionality with a NumPy backend.  If you
wish to run on accelerators or compute gradients, install the JAX
extras:

```bash
pip install flexipde[jax]
```

**Note:** JAX wheels are platform‑specific.  See the [JAX
installation guide](https://github.com/google/jax#installation) for
instructions on installing CPU or GPU builds.  Once JAX is installed
successfully, `flexipde` will automatically switch to the JAX
backend when performing simulations.

To use optimisation utilities (Optax and Flax) and pretty terminal
output, install the `optim` extra:

```bash
pip install flexipde[optim]
```

To build the documentation locally, install the `docs` extras:

```bash
pip install flexipde[docs]
```

## Installing from source

You can also clone the repository and install it in editable mode
during development:

```bash
git clone https://github.com/uwplasma/flexipde.git
cd flexipde
pip install -e .[jax,optim,docs]
```

This will install all optional dependencies and allow you to modify
the source code while immediately seeing changes when running
examples.

## Optional dependencies

| Extra       | Packages                                          | Purpose                                  |
|-------------|---------------------------------------------------|-------------------------------------------|
| `jax`       | `jax`, `equinox`, `diffrax`                        | High‑performance JAX backend and ODE solvers |
| `optim`     | `optax`, `flax`, `rich`                            | Optimisation routines and pretty output     |
| `typing`    | `jaxtyping`                                       | Static type annotations for JAX arrays     |
| `symbolic`  | `sympy2jax`                                       | Symbolic expressions compiled to JAX code  |
| `docs`      | `mkdocs`, `mkdocs-material`, `mkdocstrings-python` | Building the documentation site            |

You can mix these extras as needed.  For example, to install both
`jax` and `optim` you can run:

```bash
pip install flexipde[jax,optim]
```