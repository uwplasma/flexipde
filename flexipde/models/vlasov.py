"""
Vlasov–Poisson models for kinetic plasma physics.

This module defines a simple 1D Vlasov–Poisson solver suitable for
demonstrating kinetic effects such as the two‐stream instability.  The
equation solved is::

    ∂ₜ f(x, v, t) + v ∂ₓ f + E(x, t) ∂ᵥ f = 0,

where the self–consistent electric field ``E(x, t)`` is obtained from
Poisson’s equation::

    ∂ₓ E = ρ(x, t) - ρ₀,

with charge density ``ρ(x, t) = ∫ f(x, v, t) dv``.  The neutralising
background density ``ρ₀`` is chosen such that the zero mode of the
electric field vanishes.  We restrict ourselves to periodic boundary
conditions in space and velocity, and normalise physical constants so
that the charge and mass are unity.

The solver uses a spectral method in the spatial dimension (via the
provided discretiser) and a centred finite difference in the velocity
dimension.  The model supports both NumPy and JAX arrays; if JAX is
available and the JAX optional dependency is installed, the right hand
side will be JIT compiled automatically by Diffrax when used in a
simulation.  The initial condition is parameterised by a mixture of
Maxwellian beams separated by a drift velocity and can be customised
through the ``ic_params`` argument.

Examples
--------
See the ``vlasov.toml`` example in the ``examples`` directory for a
configuration file that runs a simple two–stream instability.  You
can also use the optimisation utilities in :mod:`flexipde.optim`
to tune parameters such as the temperature ratio to maximise the
growth rate of the instability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as _np

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _HAS_JAX = False

from .base import PDEModel
from flexipde.grid import Grid
from flexipde.discretisation.base import BaseDifferentiator


def _choose_backend(use_jax: bool) -> Any:
    """Return the appropriate numerical module given availability of JAX.

    Parameters
    ----------
    use_jax : bool
        If True and JAX is available, return ``jax.numpy``.  Otherwise
        return ``numpy``.

    Returns
    -------
    module
        Either :mod:`jax.numpy` or :mod:`numpy`.
    """
    if use_jax and _HAS_JAX:
        return jnp
    return _np


@dataclass
class VlasovTwoStream(PDEModel):
    """One–dimensional Vlasov–Poisson model for the two–stream instability.

    Parameters
    ----------
    grid : :class:`~flexipde.grid.Grid`
        Spatial grid along the x direction.  Only the first dimension
        of the grid is used; velocity space is handled internally.
    diff : :class:`~flexipde.discretisation.base.BaseDifferentiator`
        Discretisation scheme used for spatial derivatives in x.
    nv : int
        Number of velocity grid points.  A larger value increases
        resolution in velocity space but also increases computational
        cost.
    v_min : float
        Lower bound of the velocity domain.
    v_max : float
        Upper bound of the velocity domain.  The velocity grid is
        equispaced between ``v_min`` and ``v_max``.
    """
    nv: int
    v_min: float
    v_max: float

    def __post_init__(self) -> None:
        # Precompute the velocity grid and spacing.  We keep separate
        # arrays for v and dv so that we can use either numpy or jax
        # depending on the runtime environment.  ``self._v`` and
        # ``self._dv`` will be assigned on first call to
        # :meth:`initial_state` when the backend is known.
        self._v_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Backend‑agnostic helpers

    def _get_velocity_grid(self, xp: Any) -> Any:
        """Return the velocity grid and spacing for the chosen backend.

        The grid is cached separately for NumPy and JAX to avoid
        recreating it on every call.

        Parameters
        ----------
        xp : module
            Numerical module (:mod:`numpy` or :mod:`jax.numpy`).

        Returns
        -------
        tuple of (v, dv)
            1D array of velocities and scalar spacing ``dv``.
        """
        key = 'jax' if xp is jnp else 'numpy'
        if key not in self._v_cache:
            v = xp.linspace(self.v_min, self.v_max, self.nv, endpoint=False)
            dv = (self.v_max - self.v_min) / self.nv
            self._v_cache[key] = (v, dv)
        return self._v_cache[key]

    def _poisson_field(self, f: Any, xp: Any) -> Any:
        """Compute electric field ``E`` from distribution ``f``.

        Parameters
        ----------
        f : array, shape (nx, nv)
            Distribution function.
        xp : module
            Numerical backend.

        Returns
        -------
        E : array, shape (nx,)
            Electric field evaluated on the spatial grid.

        Notes
        -----
        The charge density is computed by integrating over velocity
        with weight ``dv``.  We assume a neutralising background
        density such that the zero Fourier mode of the field vanishes.
        """
        # Integrate f over velocity to get density rho(x).  Multiply by
        # dv to approximate the integral.
        v, dv = self._get_velocity_grid(xp)
        rho = xp.sum(f, axis=1) * dv
        # Remove the background density: we want only fluctuations
        # around the mean to drive the field.  Subtract the mean of
        # rho so that the k=0 Fourier mode of the potential vanishes.
        rho = rho - xp.mean(rho)
        # Solve Poisson's equation in Fourier space:
        #   d^2 phi/dx^2 = -rho
        # We compute phi_k = rho_k / (k^2) (with a minus sign from the
        # right hand side) then E_k = -i k phi_k.  The zero mode k=0
        # yields zero field.  Wavenumbers depend on the grid spacing.
        nx = f.shape[0]
        # Use the same dx as the grid spacing along x.
        # Compute spacing along x.  The Grid spacing() method returns a tuple
        # but is not a property, so call it to obtain the spacings.
        dx = self.grid.spacing()[0]
        # Compute Fourier modes of rho along x.
        rho_hat = xp.fft.fft(rho)
        # Wavenumbers: 2π k / L, with L = nx * dx.  Use fftfreq to
        # generate k / L and multiply by 2π afterwards.
        k = xp.fft.fftfreq(nx, d=dx) * (2.0 * xp.pi)
        # Avoid divide by zero for k=0.  We will set E_hat[0] = 0
        k_sq = k * k
        # Compute potential phi_hat.  Note that the sign is such that
        # phi_hat = -rho_hat / k^2.  We avoid division by zero by
        # selectively dividing only on non-zero wavenumbers.
        # For numpy we use an explicit mask; for JAX jnp.where is safe.
        if xp is _np:
            # For NumPy, avoid division by zero by explicit masking
            phi_hat = xp.zeros_like(rho_hat, dtype=complex)
            mask = k_sq != 0.0
            phi_hat[mask] = -rho_hat[mask] / k_sq[mask]
        else:
            # For JAX, jnp.where lazily selects without computing both
            # branches, so it's safe to use directly
            phi_hat = xp.where(k_sq != 0.0, -rho_hat / k_sq, 0.0)
        # Electric field in Fourier space: E_hat = -i k phi_hat
        E_hat = -1j * k * phi_hat
        # Transform back to real space
        E = xp.fft.ifft(E_hat)
        # The result should be real to numerical precision.
        return xp.real(E)

    # ------------------------------------------------------------------
    # Public API implemented for PDEModel

    def initial_state(self, ic_params: Optional[dict] = None) -> Dict[str, Any]:
        """Construct an initial distribution for the two–stream instability.

        Parameters
        ----------
        ic_params : dict, optional
            Optional dictionary specifying parameters for the initial
            condition.  The following keys are recognised:

            * ``amplitude`` (float or array, default 1e-3): initial amplitude of
              the sinusoidal density perturbation.  When using the JAX
              backend this may be a JAX scalar for differentiability.
            * ``mode`` (int, default 1): spatial mode number of the
              perturbation (perturbation is proportional to cos(2π mode x / L)).
            * ``drift_velocity`` (float, default 1.0): stream velocity ±v₀
              in units of thermal speed.
            * ``thermal_velocity`` (float, default 1.0): thermal width of
              each Maxwellian beam.
            * ``background_density`` (float, default 1.0): normalisation of
              the distribution function.
            * ``backend`` (str, optional): ``"jax"`` or ``"numpy"``.

        Returns
        -------
        dict
            Mapping ``{"f": f0}`` where ``f0`` is an array of shape
            ``(nx, nv)``.
        """
        if ic_params is None:
            ic_params = {}
        # Determine backend from ic_params or use JAX if available
        backend = ic_params.get("backend")
        use_jax = (backend == "jax") or (backend is None and _HAS_JAX)
        xp = jnp if (use_jax and _HAS_JAX) else _np  # type: ignore
        # Number of spatial points
        nx = self.grid.shape[0]
        # Spatial grid: compute domain length from spacing and number of points
        dx = self.grid.spacing()[0]
        domain_length = dx * nx
        x = xp.linspace(0.0, domain_length, nx, endpoint=False)
        # Extract parameters with defaults.  Do not convert to Python
        # floats so that JAX arrays may propagate through.  The values
        # may be Python floats, ints or JAX scalars.  We rely on JAX
        # broadcasting to handle mixed types.
        amplitude = ic_params.get("amplitude", 1e-3)
        mode = ic_params.get("mode", 1)
        v0 = ic_params.get("drift_velocity", 1.0)
        vt = ic_params.get("thermal_velocity", 1.0)
        n0 = ic_params.get("background_density", 1.0)
        # Velocity grid and spacing
        v, dv = self._get_velocity_grid(xp)
        # Maxwellian beams centred at ±v0
        f_stream = (
            xp.exp(-((v - v0) / vt) ** 2)
            + xp.exp(-((v + v0) / vt) ** 2)
        )
        # Normalise so that ∫ f dv = 2 (number density of both beams)
        norm = xp.sum(f_stream) * dv
        f_stream = 2.0 * f_stream / norm
        # Base distribution with spatially uniform density n0
        f0 = n0 * xp.broadcast_to(f_stream, (nx, self.nv))
        # Add a small sinusoidal perturbation in density.  Use xp.cos so
        # that JAX will differentiate through the perturbation if needed.
        # Convert mode to integer for numpy; JAX accepts Python ints.
        perturb = 1.0 + amplitude * xp.cos(2.0 * xp.pi * mode * x / domain_length)
        f0 = f0 * perturb[:, None]
        return {"f": f0}

    def rhs(self, state: Dict[str, Any], t: float) -> Dict[str, Any]:
        """Compute time derivative of the distribution function.

        Parameters
        ----------
        state : dict
            Current state mapping field names to arrays.  Only the
            ``"f"`` entry is used.
        t : float
            Current time (unused but included for compatibility with
            Diffrax and other integrators).

        Returns
        -------
        dict
            Mapping ``{"f": df_dt}`` where ``df_dt`` has the same
            shape as the input ``f``.
        """
        f = state["f"]
        # Determine backend from the array type
        use_jax = _HAS_JAX and isinstance(f, jax.Array)
        xp = _choose_backend(use_jax)
        # Compute electric field from f
        E = self._poisson_field(f, xp)
        # Velocity grid
        v, dv = self._get_velocity_grid(xp)
        # Compute derivative of f with respect to x using the provided
        # discretisation.  We only use the first spatial dimension of
        # the grid; the discretiser will broadcast over the velocity
        # dimension automatically.
        # We call gradient and take the derivative along axis 0.
        # Compute derivative of f with respect to x using the provided
        # discretisation.  For a 2D array ``f`` of shape (nx, nv),
        # differentiate along axis 0 (the spatial dimension).  The
        # finite difference and spectral differentiators implement
        # ``grad`` for a scalar field and will broadcast over the
        # remaining axes.  The result has the same shape as ``f``.
        dfdx = self.diff.grad(f, 0)
        # Compute derivative in velocity by centred finite differences
        # along axis 1.  Implement periodic boundaries: wrap indices.
        f_roll_forward = xp.roll(f, -1, axis=1)
        f_roll_backward = xp.roll(f, 1, axis=1)
        dfdv = (f_roll_forward - f_roll_backward) / (2.0 * dv)
        # Compute time derivative: df/dt = -v * dfdx - E * dfdv
        # We need to broadcast v over x and E over v
        dfdt = -v[None, :] * dfdx - E[:, None] * dfdv
        return {"f": dfdt}

    @property
    def output_fields(self) -> Any:
        """Return names of fields produced by the model."""
        return ["f"]