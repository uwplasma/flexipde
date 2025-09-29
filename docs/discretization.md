# Discretization Schemes

FlexiPDE separates the *model* (the set of partial differential equations to
be solved) from the *discretization* (the numerical method used to approximate
spatial derivatives).  This modular approach allows you to swap between
high–order spectral methods, finite–difference stencils, and even curvilinear
coordinates without changing the underlying physics model.

## Spectral differentiation

For periodic domains, the code provides a **spectral** discretizer.  Spatial
derivatives are computed via the Fourier transform: for a field $u(x)$ defined
on a grid $x\_j$, the derivative is computed by multiplying each Fourier mode
by its wavenumber.  For example, the first derivative of $u$ in one
dimension is

\[\frac{\partial u}{\partial x} = \mathcal{F}^{-1}\bigl( i k \cdot \hat{u}(k)\bigr),\]

where $\hat{u}(k)$ is the discrete Fourier transform of $u$.  The
implementation uses either `numpy.fft` or `jax.numpy.fft` depending on the
backend selected when constructing the discretizer.  The Laplacian
$\nabla^2 u$ is computed similarly by multiplying the Fourier modes by
$(i k)^2$ along each axis.

Spectral differentiation is exact for smooth periodic functions and exhibits
exponential convergence as you refine the grid.  It is the default choice
when the domain is periodic in all directions and you need high accuracy with
few grid points.

## Finite–difference schemes

For non‑periodic domains or problems where you need boundary conditions
explicitly enforced, FlexiPDE provides a **finite–difference** discretizer.
Central differences of second order are implemented for the gradient and
Laplacian operators.  Higher–order stencils can be added easily by
subclassing the base discretizer.

Given a uniform grid $x\_j$ with spacing $h$, the second–order central
difference approximation to the first derivative is

\[\frac{\partial u}{\partial x}(x\_j) \approx \frac{u\_{j+1} - u\_{j-1}}{2h},\]

and the one–dimensional Laplacian is

\[\frac{\partial^2 u}{\partial x^2}(x\_j) \approx \frac{u\_{j+1} - 2u\_{j} + u\_{j-1}}{h^2}.\]

At boundaries, one–sided differences are used according to the specified
boundary condition (Dirichlet, Neumann or periodic).  The finite difference
discretizer works for multidimensional problems by applying the stencil along
each axis in turn.  It supports curvilinear coordinates by taking metric
factors into account when computing derivatives and the Laplacian.

## Custom discretizers

You can implement your own discretization by subclassing
`flexipde.discretisation.BaseDifferentiator` and implementing the
`grad`, `divergence` and `laplacian` methods.  See the existing
`SpectralDifferentiator` and `FiniteDifference` classes for reference.