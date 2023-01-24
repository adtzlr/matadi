import casadi

from .__about__ import __version__

Variable = casadi.SX.sym

from . import math, models
from ._lab_compressible import LabCompressible
from ._lab_compressible import LabCompressible as Lab
from ._lab_incompressible import LabIncompressible
from ._material import Function
from ._material import Function as FunctionScalar
from ._material import FunctionTensor
from ._material import Material
from ._material import Material as MaterialScalar
from ._material import MaterialTensor
from ._templates import (
    MaterialComposite,
    MaterialHyperelastic,
    MaterialHyperelasticPlaneStrain,
    MaterialHyperelasticPlaneStressIncompressible,
    MaterialHyperelasticPlaneStressLinearElastic,
    MaterialTensorGeneral,
    ThreeFieldVariation,
    ThreeFieldVariationPlaneStrain,
    TwoFieldVariation,
    TwoFieldVariationPlaneStrain,
)

__all__ = [
    "__version__",
]
