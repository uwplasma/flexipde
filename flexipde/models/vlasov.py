r"""1D Vlasov–Poisson two‑stream instability.

This model solves a 1D Vlasov equation coupled to Poisson's equation for
the electric field, modelling the two‑stream instability.  The system is

.. math::

    \partial_t f + v \partial_x f + E \partial_v f = 0,
    \partial_x E = 1 - \int f \, dv.

Periodic boundary conditions are used in ``x`` and vanishing fields at
``|v| \to \infty``.  The solver uses a spectral derivative in ``x`` and
a finite difference in ``v``.  This implementation is intended for
educational purposes and is not optimised.
"""

# mypy: ignore-errors
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as _np

try:
    import jax.numpy as _jnp  # type: ignore[attr-defined]
    _JAX_AVAILABLE = True
except Exception:
    _jnp = None
    _JAX_AVAILABLE = False

from ..grid import Grid
from ..discretisation.spectral import SpectralDifferentiator
from .base import PDEModel


@dataclass
class VlasovTwoStream(PDEModel):
    """Two‑stream Vlasov–Poisson model.

    Parameters
    ----------
    grid:
        The spatial grid (1D).
    diff:
        The spectral differentiator for the spatial derivative.  The model
        requires periodic boundaries.
    nv:
        Number of velocity grid points.
    v_min, v_max:
        Velocity domain limits.
    amplitude:
        Amplitude of the initial perturbation.
    drift_velocity:
        Mean drift velocity of the two streams.
    thermal_velocity:
        Thermal velocity spread of the Maxwellian.
    background_density:
        Total background density (normalisation constant).  The equilibrium
        distribution is ``(background_density/2) * [exp(-(v - v0)^2 / (2 vt^2)) + exp(-(v + v0)^2 / (2 vt^2))]``.
    """

    nv: int = 64
    v_min: float = -5.0
    v_max: float = 5.0
    amplitude: float = 0.05
    drift_velocity: float = 2.0
    thermal_velocity: float = 1.0
    background_density: float = 1.0

    def __post_init__(self) -> None:
        # Check diff is spectral
        if not isinstance(self.diff, SpectralDifferentiator):
            raise ValueError("VlasovTwoStream requires a spectral differentiator in x")
        # Velocity grid and spacing
        self.v_grid = _np.linspace(self.v_min, self.v_max, self.nv)
        self.dv = (self.v_max - self.v_min) / (self.nv - 1)
        self.field_names = ["f"]
        super().__post_init__()

        # Precompute k^2 and i*k for Poisson solver in x
        # Use diff._k_axes for x dimension (only one)
        k = self.diff._k_axes[0]  # type: ignore[attr-defined]
        xp = self.diff._xp  # type: ignore[attr-defined]
        self._k_sq = k * k
        # Avoid division by zero for k=0 mode (set potential to zero mean)
        self._inv_k_sq = xp.where(self._k_sq != 0, 1.0 / self._k_sq, 0.0)

    def initial_state(self, ic_params: dict[str, Any] | None = None) -> dict[str, Any]:
        # Create equilibrium distribution: sum of two Maxwellians
        xp = _jnp if (self.diff._backend == "jax" and _JAX_AVAILABLE) else _np  # type: ignore[attr-defined]
        x_coords = self.grid.coords[0]
        mesh_x, mesh_v = xp.meshgrid(x_coords, self.v_grid, indexing='ij')
        vt = self.thermal_velocity
        v0 = self.drift_velocity
        f0 = 0.5 * self.background_density * (
            xp.exp(-((mesh_v - v0) ** 2) / (2.0 * vt ** 2))
            + xp.exp(-((mesh_v + v0) ** 2) / (2.0 * vt ** 2))
        ) / (vt * xp.sqrt(2.0 * xp.pi))
        # Add sinusoidal perturbation in x
        perturb = 1.0 + self.amplitude * xp.cos(2.0 * xp.pi * mesh_x / (x_coords[-1] - x_coords[0]))
        f = f0 * perturb
        return {"f": f}

    def _electric_field(self, f: Any) -> Any:
        """Compute the electric field from the charge density by solving Poisson's equation."""
        xp = self.diff._xp  # type: ignore[attr-defined]
        # Integrate f over v to get density
        rho = xp.trapz(f, self.v_grid, axis=-1)
        # Normalise: background density is unity
        rho = self.background_density - rho
        # Solve Poisson: phi_hat = -rho_hat / k^2, E_hat = i k phi_hat
        rho_hat = self.diff._fft(rho)  # type: ignore[attr-defined]
        phi_hat = -rho_hat * self._inv_k_sq
        # Electric field E_x = -d phi/dx => multiply by i k
        k = self.diff._k_axes[0]  # type: ignore[attr-defined]
        shape = [1] * phi_hat.ndim
        shape[0] = k.size
        k_reshaped = k.reshape(shape)
        E_hat = -1j * k_reshaped * phi_hat
        E = self.diff._ifft(E_hat)  # type: ignore[attr-defined]
        return xp.real(E)

    def rhs(self, state: dict[str, Any], t: Any) -> dict[str, Any]:
        f = state["f"]
        xp = self.diff._xp  # type: ignore[attr-defined]
        # Spatial derivative
        df_dx = self.diff.grad(f)[0]
        # Velocity derivative (central difference)
        forward = xp.roll(f, -1, axis=-1)
        backward = xp.roll(f, 1, axis=-1)
        df_dv = (forward - backward) / (2.0 * self.dv)
        # One‑sided boundaries
        df_dv = df_dv.copy()
        df_dv[..., 0] = (f[..., 1] - f[..., 0]) / self.dv
        df_dv[..., -1] = (f[..., -1] - f[..., -2]) / self.dv
        # Electric field
        E = self._electric_field(f)
        rhs_f = - (self.v_grid[None, :] * df_dx + E[..., None] * df_dv)
        return {"f": rhs_f}