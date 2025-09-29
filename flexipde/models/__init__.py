"""Predefined PDE models.

The models subpackage contains classes that define the right‑hand side of
partial differential equations.  Each model subclasses
:class:`~flexipde.models.base.PDEModel` and specifies the set of fields,
parameters and the function that computes time derivatives.

See the documentation for examples and derivations of the equations.
"""

from .base import PDEModel
from .advection import LinearAdvection

# Backwards compatibility alias.  The simpler name ``Advection`` can be used
# instead of ``LinearAdvection`` in configuration files and examples.
Advection = LinearAdvection
from .diffusion import Diffusion
from .resistive_mhd import ResistiveMHD
from .two_fluid import TwoFluid
from .drift_kinetic import DriftKinetic
from .ideal_alfven import IdealAlfven
from .vlasov import VlasovTwoStream

__all__ = [
    "PDEModel",
    "LinearAdvection",
    "Advection",
    "Diffusion",
    "ResistiveMHD",
    "TwoFluid",
    "DriftKinetic",
    "IdealAlfven",
    "VlasovTwoStream",
]