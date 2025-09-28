# Built‑in models

`flexipde` ships with several models representing common equations in
plasma physics.  Each model subclasses `PDEModel` and implements a
method `initial_state` to generate initial conditions and a method
`rhs` to compute the time derivative of the fields.  This section
describes the models included in the core package.  You can use
them directly or as templates for your own models.

## Advection

The **linear advection** model transports a scalar field `u(x, t)`
with a constant velocity vector **v**:

\[ \partial_t u + \boldsymbol{v}\cdot\nabla u = 0. \]

Because the velocity is constant, this equation has the analytical
solution `u(x, t) = u_0(x - v t)`.  It is often used as a
verification test for numerical methods.  The model accepts a
velocity list of length equal to the grid dimension.  Initial
conditions can be Gaussian, sinusoidal or constant; you can also
provide your own function via the `init_u` argument.

### Parameters

* `velocity` – sequence of floats giving the velocity components.
* `init_u` – optional callable returning an array of shape equal to the grid.  If supplied, `ic_params` must be `None`.

### Initial condition keys

* `type` – `"gaussian"`, `"sinusoidal"` or `"constant"`.
* `amplitude` – amplitude of the Gaussian or sinusoid.
* `wavevector` – integer wave numbers along each dimension for the sinusoid.
* `phase` – phase shift (radians) applied to the sinusoid.
* `value` – constant value for the `"constant"` type.

## Diffusion

The **diffusion** model solves the heat equation for a scalar field
`u(x, t)`:

\[ \partial_t u = D \nabla^2 u. \]

where `D` is a constant diffusivity.  Periodic or non‑periodic
boundary conditions can be specified via the grid.  The initial
condition keys are the same as for the advection model.

### Parameters

* `diffusivity` – scalar diffusivity `D`.

## Ideal Alfvén waves

The **IdealAlfven** model implements a simplified version of ideal
magnetohydrodynamics (MHD) describing transverse Alfvén waves on a
uniform background magnetic field **B₀**.  In 1D the coupled
equations for the transverse velocity `v_y(x, t)` and magnetic field
`B_y(x, t)` are

\[ \partial_t v_y = \partial_x B_y, \quad \partial_t B_y = \partial_x v_y. \]

These equations support waves propagating at the Alfvén speed (taken
to be unity here).  The model accepts a list `B0` giving the
background magnetic field components.  Initial conditions should
specify arrays for `v` and `B` of the same shape as the grid.

### Parameters

* `B0` – sequence of floats specifying the background field.  The
  length must match the number of spatial dimensions.

## Vlasov–Poisson (two–stream instability)

The **VlasovTwoStream** model solves a 1D Vlasov–Poisson system
coupled to an electrostatic field.  The evolution equation for the
distribution function `f(x, v, t)` is

\[ \partial_t f + v\,\partial_x f + E(x, t)\,\partial_v f = 0, \]

and Poisson’s equation determines the electric field `E(x, t)` from
the charge density `\rho(x) = \int f(x, v)\,dv` via

\[ \partial_x E = \rho(x) - \langle \rho \rangle, \]

where angle brackets denote the spatial average.  The model uses a
spectral solver in the spatial dimension and centred finite
differences in the velocity dimension.  It supports both NumPy and
JAX backends.  The two–stream instability is obtained by choosing an
initial distribution consisting of two Maxwellian beams centred at
±`drift_velocity`.  The perturbation amplitude and mode number
control the initial density perturbation.

### Parameters

* `nv` – number of velocity grid points.
* `v_min`, `v_max` – velocity domain bounds.

### Initial condition keys

* `amplitude` – amplitude of the sinusoidal density perturbation.
* `mode` – spatial mode number of the perturbation.
* `drift_velocity` – beam velocity ±v₀.
* `thermal_velocity` – thermal width of each beam.
* `background_density` – overall normalisation of the distribution.

## Extending models

To create your own model, subclass `PDEModel` and implement
`initial_state(ic_params)` and `rhs(state, t)`.  The `grid` and
`diff` objects attached to the model can be used to compute spatial
derivatives.  The `apply_boundary_conditions` method enforces
Dirichlet or Neumann boundaries.  For functional or data‑driven
models, follow the design of JAX‑MD by representing state as arrays
and writing pure functions for the right‑hand side【925522637824375†L37-L40】.