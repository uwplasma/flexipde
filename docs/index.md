% flexipde: A flexible, differentiable plasma physics solver
% =========================================================

`flexipde` is an open‑source library for simulating a wide range of
plasma physics equations in Python.  It is designed to be both
**flexible** – new models, boundary conditions and discretisation
methods can be added with ease – and **high‑performance**, using
[JAX](https://github.com/google/jax) for just‑in‑time compilation,
automatic differentiation and hardware acceleration on CPUs, GPUs and
TPUs.  The goal is to provide a unified framework that spans
**fluid**, **magnetohydrodynamic (MHD)** and **kinetic** models, and
scales from desktop machines to high‑performance clusters.

### Why flexipde?

Research codes like **BOUT++** and **Φ‑Flow** demonstrate the power
of modular design.  BOUT++ decouples geometry from physics so that
users can evolve *any number of equations* in arbitrary curvilinear
coordinates using a variety of numerical methods【501252340464299†L24-L33】.  Φ‑Flow
provides an object‑oriented API that runs the **same code in 2D or
3D on different backends** (NumPy, PyTorch, TensorFlow, JAX)【383486560010699†L364-L376】.
flexipde builds on these ideas while taking advantage of JAX’s
automatic differentiation and vectorised execution.  Our guiding
principles are:

* **Modularity** – separate geometry (the grid) from discretisation
  and from the physics model.  This mirrors the BOUT++ philosophy of
  keeping geometry independent of the equations【501252340464299†L24-L33】.
* **Multi‑back‑end execution** – allow the same code to run with
  either NumPy (for quick prototyping) or JAX (for performance and
  auto‑diff).  Φ‑Flow demonstrated that such code reuse is possible
  across dimensions and backends【383486560010699†L364-L376】.
* **Differentiable physics** – treat the solver as a pure function of
  its inputs so that gradients of outputs with respect to initial
  conditions and parameters can be computed using `jax.grad` and
  Diffrax’s adjoint methods【805256974599970†L81-L91】.  JAX‑Fluids shows how
  fully differentiable solvers unlock new research directions such as
  learning closure models【889881592370340†L35-L41】.
* **High‑performance computing** – use JAX sharding and JIT
  compilation to automatically parallelise over multiple CPU cores or
  GPUs.  Large‑scale runs can therefore exploit cluster resources
  without changing user code.

### Key capabilities

* **Multiple physics models.**  Built‑in models include linear
  advection, diffusion, ideal Alfvén waves (a simplified MHD model)
  and a Vlasov–Poisson solver for the two–stream instability.  New
  models can be implemented by subclassing `PDEModel` and defining
  the initial state and right‑hand side.
* **Flexible discretisation.**  Choose between Fourier spectral
  differentiation (for periodic domains) and finite differences.  Grid
  coordinates and metrics are stored in a `Grid` object, which can be
  constructed from a simple configuration dictionary.  Curvilinear
  coordinates are supported via user‑supplied metric tensors.
* **Declarative configuration.**  Simulations can be run from a
  single TOML file specifying the grid, discretisation, model
  parameters, solver settings and initial conditions.  Batch runs
  over multiple initial conditions are supported out of the box.
* **Differentiable and optimisable.**  With JAX and Diffrax
  installed, the solver is fully differentiable.  The `flexipde.optim`
  module provides functions to compute gradients of scalar objectives
  and run Optax optimisers, enabling inverse problems and
  physics‑informed learning.
* **HPC ready.**  When run with JAX, simulations are JIT compiled and
  automatically parallelised across devices.  Vectorised batches of
  initial conditions are solved in parallel using `jax.vmap` and
  Diffrax’s support for batched integration.

To get started, follow the [Installation](installation.md) instructions
and see the [User guide](user_guide.md) for examples.