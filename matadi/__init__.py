from .__about__ import __version__

import casadi

Variable = casadi.SX.sym

from ._material import (
    Function,
    Function as FunctionScalar,
    FunctionTensor,
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
from ._lab_compressible import (
    LabCompressible,
    LabCompressible as Lab,
)
from ._lab_incompressible import (
    LabIncompressible,
)


__all__ = [
    "__version__",
]
