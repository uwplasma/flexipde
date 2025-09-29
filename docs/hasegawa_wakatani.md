---
title: Hasegawa–Wakatani model
---

# Hasegawa–Wakatani model

The Hasegawa–Wakatani (HW) system is a reduced model of drift–wave turbulence in magnetised plasmas.  It couples the density perturbation $n$ and electrostatic potential $\phi$ through advection and parallel diffusion【820049995294040†L22-L78】.  In two dimensions, ignoring curvature effects, the equations can be written as

$$
\partial_t n + \{\phi,n\} + \alpha (\phi - n) = \kappa \partial_{y} \phi,
$$

$$
\partial_t \phi + \{\phi,\phi\} + \alpha (\phi - n) = \kappa \partial_{y} n,
$$

where $\{\cdot,\cdot\}$ denotes the Poisson bracket $\{f,g\}=\partial_x f\,\partial_y g - \partial_y f\,\partial_x g$, $\alpha$ measures the strength of parallel electron dynamics, and $\kappa$ drives diamagnetic drifts【820049995294040†L22-L78】.

flexipde does not include the HW model as a built‑in class, but it is straightforward to implement it in a custom script.  See `examples/run_hasegawa_wakatani.py` for a working implementation that constructs a 2D grid, uses the spectral differentiator for derivatives, defines a `rhs` function computing the Poisson bracket and source terms, and integrates the equations in time.  Because the equations are non‑linear, you should use JAX and Diffrax for efficient integration and automatic differentiation.

## Manufactured solution test

To verify that the HW implementation is correct, you can use the method of manufactured solutions.  For example, choose

$$
n(x,y,t) = \cos(2x)\cos(3t), \quad \phi(x,y,t) = \sin(2y)\sin(3t).
$$

Substitute these into the HW equations to compute the required forcing terms, then implement the `rhs` function accordingly.  A test in `tests/test_manufactured.py` shows how to implement such checks for simpler models.
