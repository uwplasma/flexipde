---
title: Installation
---

# Installation

flexipde is available on PyPI.  You can install the core package with:

```bash
pip install flexipde
```

Optional dependencies can be pulled in via extras:

* **JAX support:** `pip install 'flexipde[jax]'` installs JAX, Diffrax, Equinox and Optax for GPU acceleration and automatic differentiation.
* **Documentation:** `pip install 'flexipde[docs]'` installs MkDocs Material and mkdocstrings to build the documentation locally.
* **Testing:** `pip install 'flexipde[testing]'` installs pytest, pytest‑cov and mypy for running the test suite and static analysis.

To install from source (for development), clone the repository and install in editable mode:

```bash
git clone https://github.com/uwplasma/flexipde.git
cd flexipde
pip install -e .[jax,testing,docs]
```

The library is pure Python and requires only NumPy by default.  On Python 3.11 and newer, the standard library `tomllib` is used to parse configuration files; on older Pythons, the `tomli` backport is required and pulled in automatically.