# Examples

This section describes the example configuration files and scripts
provided with `flexipde`.  You can find them in the
`examples/` directory of the repository.  Each example can be run
either from Python or via the command line using the module
`flexipde.run`.

## advection.toml

Simulates advection of a sine wave around a periodic domain.  The
configuration specifies a 1D domain on `[0, 2π)`, uses the spectral
discretisation with the JAX backend, and advects the field with
velocity 1.  The initial condition is a sine wave of mode 1.  Run it
with:

```bash
python -m flexipde.run examples/advection.toml --output advection.npz
```

The result file will contain arrays `u` and `times`.  You can plot
the initial and final profiles to verify that the wave has shifted
without distortion.

## alfven.toml

Demonstrates the IdealAlfven model by propagating a transverse
velocity and magnetic field perturbation on a 1D domain.  The
background field is `B0 = [1.0]` and the perturbation is a sine
wave.  The spectral discretisation is used.  After one Alfvén
period the perturbations return to their initial shape, illustrating
the nondispersive nature of Alfvén waves.

## vlasov.toml

Runs the Vlasov–Poisson two–stream instability.  The configuration
defines a 1D spatial grid, a velocity grid of 64 points from
`v_min = -5` to `v_max = 5`, and sets the initial condition to two
Maxwellian beams with a small sinusoidal perturbation.  The spectral
scheme in `x` and centred differences in `v` are used.  To run:

```bash
python -m flexipde.run examples/vlasov.toml --output vlasov.pkl.gz
```

The output is saved as a gzipped pickle containing the `SimulationResult`.
You can reload it via `flexipde.result.SimulationResult.load` and
analyse the distribution `f` and electric field energy.

## optimise_vlasov_growth.py

This Python script shows how to combine the Vlasov model with
optimisation.  It defines an objective equal to the negative of the
electric field energy at the end of the simulation and uses
gradient ascent to adjust the thermal velocity such that the growth
rate of the two–stream instability is maximised.  The script relies
on the optimisation utilities in `flexipde.optim` and requires
JAX, Diffrax and Optax.

To run:

```bash
python examples/optimise_vlasov_growth.py
```

The script prints the optimal thermal velocity and final loss.

## run\_diffusion\_1d.py

Solves the 1D diffusion equation on a non‑periodic domain with
Neumann boundary conditions using a finite‑difference discretisation.
A Gaussian initial profile centred in the domain diffuses and
flattens over time.  At the end of the run the script reports the
minimum and maximum values of the solution and plots the initial and
final profiles for comparison.  Run it with:

```bash
python examples/run_diffusion_1d.py
```

## run\_diffusion\_2d.py

Simulates diffusion on a 2D square domain with Dirichlet boundary
conditions (fixed values) using finite differences.  The initial
condition is a Gaussian bump in the centre of the domain which
spreads out as time progresses.  The script displays contour plots
of the initial and final states so you can visually assess the
diffusion process.  Run it with:

```bash
python examples/run_diffusion_2d.py
```

## run\_diffusion\_3d.py

Demonstrates 3D diffusion on a periodic cube using a spectral
discretisation.  A 3D Gaussian blob is initialised and allowed to
diffuse; the script prints summary statistics and produces a 2D
slice through the domain to illustrate the decay.  Because the
spectral method assumes periodic boundaries, the solution wraps
around the edges.  Run it with:

```bash
python examples/run_diffusion_3d.py
```

## run\_cylindrical\_diffusion.py

Shows how to implement a custom diffusion model in axisymmetric
cylindrical $(r,z)$ coordinates by subclassing :class:`~flexipde.models.base.PDEModel`.
The Laplacian includes a $(1/r)\partial u/\partial r$ term which is
handled explicitly.  A Gaussian peak at the origin diffuses over
time.  The script plots the radial profile at $z=0.5$ for the
initial and final state.  Run it with:

```bash
python examples/run_cylindrical_diffusion.py
```

## optimize\_transport\_equation.py

Uses the optimisation utilities in :mod:`flexipde.optim` to tune the
parameter in a 1D transport equation $\partial_t u + \partial_x(a(x,p)
u) = 0$ where $a(x,p) = 1 + p\sin x$.  The goal is to minimise the
mean squared difference between the final and initial state, i.e.
drive the system towards a steady state.  The script iteratively
updates the parameter using Optax gradient descent and prints the
loss and parameter value at each iteration.  Run it with:

```bash
python examples/optimize_transport_equation.py
```

---

Feel free to experiment with these examples by changing the grid
resolution, discretisation scheme, initial condition parameters and
solver settings.  They serve both as demonstrations of the library
and as starting points for your own research simulations.