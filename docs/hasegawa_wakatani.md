# Hasegawa–Wakatani system

The **Hasegawa–Wakatani** (HW) model describes resistive drift–wave
turbulence in magnetised plasmas.  It couples the electron density
fluctuation $n$ to the electrostatic potential $\phi$ via a modified
continuity equation and a vorticity equation:

\[
\begin{aligned}
\frac{\partial n}{\partial t} + [\phi,n] &= -\kappa \frac{\partial \phi}{\partial y} - \alpha (\phi - n),\\
\frac{\partial \nabla^2 \phi}{\partial t} + [\phi, \nabla^2 \phi] &= -\alpha (\phi - n),
\end{aligned}
\]

where $[f,g] = \partial\_x f \partial\_y g - \partial\_y f \partial\_x g$ is the
canonical Poisson bracket, $\alpha$ is the adiabaticity parameter, and
$\kappa$ sets the background density gradient.  In the limit $\alpha\to 0$
the system reduces to the **Hasegawa–Mima** model, while $\alpha\to \infty$
gives a simple drift–wave equation.

FlexiPDE does not include a built–in HW model, but it is straightforward
to implement it as a custom model in a few lines of code.  See
`examples/run_hasegawa_wakatani.py` for an implementation that discretises
the Poisson bracket with central differences and uses a spectral solver for
the Laplacian.  The custom model inherits from the base `PDEModel` and
implements its own `rhs` and `initial_state` methods.  The example also
demonstrates how to visualise the density and potential after a short
simulation.

For more details on the Hasegawa–Wakatani system and its physical
interpretation, see the notes by Ammar Hakim【820049995294040†L22-L78】.