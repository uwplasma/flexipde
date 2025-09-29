---
title: Discretisation
---

# Discretisation methods

The accuracy and efficiency of a PDE solver depend heavily on how derivatives are computed.  flexipde provides two complementary discretisation backends and is designed to allow others to be added easily.

## Fourier spectral methods

Spectral methods are extremely accurate for smooth, periodic functions.  The spatial domain is discretised on an evenly spaced grid and the field values are transformed into Fourier space via the FFT.  Derivatives are computed by multiplying each Fourier mode by an appropriate factor (e.g. \(ik\) for a first derivative).  The inverse FFT then returns to physical space.

flexipde implements multidimensional FFTs via NumPy (or JAX if available).  The :class:`flexipde.discretisation.SpectralDifferentiator` class handles gradient, divergence and Laplacian operators in an arbitrary number of dimensions.  In non‑Cartesian coordinates you can supply metric factors which are inserted into the derivative formulas.

Because spectral methods assume periodic boundary conditions, they are best suited to domains like tori or slabs with periodic ends.  For non‑periodic boundaries you can choose a finite difference method instead.

## Finite difference methods

For non‑periodic domains, flexipde offers a simple finite difference backend, :class:`flexipde.discretisation.FiniteDifference`.  It uses central differences of second order in the interior and one‑sided differences at boundaries.  The spacing \(\Delta x\) is determined from the grid coordinates.  Higher‑order schemes (e.g. fourth‑order central differences) and advanced finite volume methods (e.g. WENO) can be added by implementing new differentiator classes.

## Metrics and curvilinear coordinates

To solve equations in curvilinear coordinates (e.g. cylindrical or spherical), provide the scale factors as the `metric` argument to the :class:`flexipde.grid.Grid`.  Each metric component multiplies the derivative operator; for example, in cylindrical coordinates the Laplacian of a scalar \(u(r,\theta,z)\) can be written

\[\nabla^2 u = \frac{1}{r} \partial_r(r \partial_r u) + \frac{1}{r^2} \partial_{\theta\theta} u + \partial_{zz} u.\]

The grid and differentiator handle these factors automatically.  See the cylindrical diffusion example in the documentation for a working case.

## Adding new methods

To implement a new discretisation, subclass :class:`flexipde.discretisation.BaseDifferentiator` and implement the `grad`, `divergence` and `laplacian` methods.  The :class:`flexipde.grid.Grid` supplies coordinate arrays and metric factors.  Once implemented, your differentiator can be used interchangeably with the built‑in ones.