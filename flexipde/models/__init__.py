"""
PDE model implementations available in :mod:`flexipde`.

This package exposes a collection of models representing a variety of
partial differential equations and dynamical systems.  Users can
import these directly from :mod:`flexipde.models` for convenience:

.. code-block:: python

    from flexipde.models import Advection, Diffusion, IdealAlfven, VlasovTwoStream

Each model subclasses :class:`~flexipde.models.base.PDEModel` and
implements the required methods :meth:`initial_state` and
 :meth:`rhs`.  Consult the documentation of each model for details
about parameters and behaviour.
"""

from .advection import LinearAdvection as Advection
from .diffusion import Diffusion
from .ideal_mhd import IdealAlfven
from .vlasov import VlasovTwoStream

__all__ = [
    "Advection",
    "Diffusion",
    "IdealAlfven",
    "VlasovTwoStream",
]