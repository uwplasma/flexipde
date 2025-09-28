# Discretisation schemes

Numerical discretisation is at the heart of any PDE solver.  In
`flexipde` the discretisation schemes are decoupled from the physics
and from the grid; you can swap one scheme for another without
changing the model.  This design mirrors the modularity of BOUT++,
which allows users to swap numerical methods at runtime【501252340464299†L24-L33】.

## Spectral differentiation

The **SpectralDifferentiator** uses the Fast Fourier Transform (FFT)
to compute derivatives exactly for periodic functions.  For a field
`u(x)` defined on a uniform grid, the Fourier transform `û_k` is
multiplied by `(i k)` and transformed back to real space to obtain
`\partial_x u`.  In `flexipde` this is handled internally by
precomputing the wavenumbers `k` from the grid spacing and shape and
applying FFTs along the appropriate axis.  Spectral methods are
highly accurate and particularly well suited to simulating waves and
plasma instabilities.

When running with the JAX backend, `SpectralDifferentiator` uses
`jax.numpy.fft` for transforms and is fully differentiable and
parallelisable.

## Finite differences

The **FiniteDifference** scheme implements second‑order centred
finite differences for first derivatives.  Given a uniform grid
spacing `dx`, the derivative of `u` with respect to `x` at index `i`
is approximated by

\[ \left. \frac{\partial u}{\partial x} \right|_{i}
   \approx \frac{u_{i+1} - u_{i-1}}{2\,\mathrm{d}x}. \]

At the domain boundaries, one‑sided differences enforce Dirichlet or
Neumann boundary conditions depending on the field.  Finite
differences can be used on non‑periodic domains and are less
expensive than spectral methods, though they introduce numerical
dispersion for wave problems.

## Metrics and curvilinear coordinates

The `Grid` class stores an optional metric tensor that allows
simulations in curvilinear coordinates.  For example, a cylindrical
geometry might use a metric with diagonal entries `(1, r^2, 1)`.  The
discretisation schemes use this metric to modify derivatives
appropriately.  In future versions we plan to add higher‑order finite
volume methods, Riemann solvers and weighted essentially non‑oscillatory
(WENO) schemes inspired by the capabilities of JAX‑Fluids【984053868268960†L31-L59】.

## Choosing a scheme

Which discretisation to use depends on your problem:

* **Periodic, smooth solutions:** Use the spectral scheme.  It
  provides spectral accuracy and is simple to configure.
* **Non‑periodic boundaries or discontinuities:** Use the finite
  difference scheme.  It respects boundary conditions and is more
  robust for shocks and steep gradients.

You can switch between schemes in a configuration file by setting
`[discretisation].scheme` to `"spectral"` or `"finite_difference"`.