# Examples

The ``examples`` directory contains ready‑to‑run configuration files
(``.toml``) and Python scripts demonstrating how to use flexipde.  To
run a TOML example from the command line, omit the ``.toml`` extension::

    flexipde examples/diffusion_1d

To run the same simulation in Python use::

    from flexipde.io import build_simulation
    sim = build_simulation('examples/diffusion_1d.toml')
    result = sim.run()

Below is a summary of the provided examples.

## Diffusion

- **diffusion_1d.toml**: 1D diffusion with Neumann boundaries.
- **diffusion_2d.toml**: 2D diffusion on a square domain with Dirichlet boundaries.
- **diffusion_3d.toml**: 3D diffusion with periodic boundaries.
- **cylindrical_diffusion.toml**: 2D diffusion in cylindrical ``(r,θ)`` coordinates.

Each of these has a corresponding ``run_diffusion_Xd.py`` script that sets up
the simulation manually and plots the result.

## Waves and MHD

- **alfven.toml**: Ideal Alfvén wave simulation.
- **resistive_mhd.toml**: Resistive MHD with simple initial conditions.

## Kinetic equations

- **vlasov.toml**: Two‑stream Vlasov–Poisson solver.
- **drift_kinetic.toml**: Simplified drift–kinetic phase‑space evolution.

## Custom models

- **run_burgers_2d_custom.py**: Defines a 2D Burgers' equation model on the fly
  using finite differences and solves it.
- **run_hasegawa_wakatani.py**: User‑implemented Hasegawa–Wakatani system.
- **run_spherical_advection_custom.py**: Solves a 2D advection equation on
  the surface of a sphere using spherical coordinates (\theta,\,\phi).  This
  example demonstrates how to supply your own coordinate system and
  discretisation to flexipde.

## Optimisation

- **optimise_vlasov_growth.py**: Optimises the thermal velocity in a two‑stream
  Vlasov simulation to minimise the growth rate.
- **optimize_transport_equation.py**: Searches for a parameter in a 1D
  transport equation to minimise time variation of the solution.