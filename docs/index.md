# Welcome to flexipde

**flexipde** is a flexible framework for solving plasma physics equations using Python.  It
supports multiple spatial discretisation schemes (spectral and finite difference),
a variety of built‑in models (advection, diffusion, resistive MHD, two‑fluid, drift‑kinetic, Alfvén waves and Vlasov–Poisson), and can be extended easily
to custom models.  The solver automatically runs on CPUs or GPUs via JAX
and can compute gradients for parameter optimisation.

This documentation is organised as follows:

- [User guide](user_guide.md) – introduction to the grid, discretisation and models.
- [Model reference](models.md) – mathematical details and implementation notes for each model.
- [Examples](examples.md) – run simulations from TOML files or Python scripts.
- [API reference](reference.md) – auto‑generated API documentation.

Enjoy exploring and contributing to flexipde!