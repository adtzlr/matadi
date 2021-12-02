from .__about__ import __version__

import casadi

Variable = casadi.SX.sym

from ._material import (
    Material,
    Material as MaterialScalar,
    MaterialTensor,
)
from . import models
from . import math
from ._templates import (
    ThreeFieldVariation,
    ThreeFieldVariationPlaneStrain,
    MaterialHyperelastic,
    MaterialComposite,
    MaterialHyperelasticPlaneStrain,
    MaterialHyperelasticPlaneStressIncompressible,
    MaterialHyperelasticPlaneStressLinearElastic,
)
from ._lab import Lab

__all__ = [
    "__version__",
]
