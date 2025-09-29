# Model reference

This page summarises the built‑in PDE models in flexipde.  Each model
inherits from :class:`flexipde.models.base.PDEModel` and implements a
``rhs`` method returning the time derivatives of its fields.

## LinearAdvection

Advection of a scalar field ``u`` with constant velocity ``v``:

.. math::
    \partial_t u + \sum_i v_i \partial_{x_i} u = 0.

Parameters:

- ``velocity``: list of floats specifying the advection velocity in each dimension.

## Diffusion

Heat equation for a scalar field ``u`` with diffusivity ``D``:

.. math::
    \partial_t u = D \nabla^2 u.

Parameters:

- ``diffusivity``: diffusion coefficient.

## ResistiveMHD

Toy model of resistive magnetohydrodynamics in 1D, evolving transverse
velocity ``v`` and magnetic field ``B``:

.. math::
    \partial_t v = \partial_x B, \qquad \partial_t B = \partial_x v + \eta \nabla^2 B.

Parameters:

- ``eta``: resistivity.

## TwoFluid

Simplified two‑fluid model where ion and electron densities advect with
prescribed velocities ``v_i`` and ``v_e``:

.. math::
    \partial_t n_s + \sum_i v_{s,i} \partial_{x_i} n_s = 0,\qquad s \in \{i,e\}.

Parameters:

- ``velocities``: list of two lists giving velocities for ions and electrons.

## DriftKinetic

Simplified drift–kinetic equation in 1D phase space without self‑consistency:

.. math::
    \partial_t f + v \partial_x f + E \partial_v f = 0.

Parameters:

- ``nv``: number of velocity grid points.
- ``v_min``, ``v_max``: velocity range.
- ``E``: constant electric field.

## IdealAlfven

Toy model of shear Alfvén waves in 1D, evolving ``v`` and ``B`` according to

.. math::
    \partial_t v = \partial_x B, \qquad \partial_t B = \partial_x v.

No parameters.

## VlasovTwoStream

1D Vlasov–Poisson solver modelling the two‑stream instability with Maxwellian
streams.  See the code for details.