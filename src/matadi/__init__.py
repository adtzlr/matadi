from . import math, models
from .__about__ import __version__
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
from ._variable import Variable

__all__ = [
    "__version__",
    "math",
    "models",
    "LabCompressible",
    "LabIncompressible",
    "Lab",
    "Function",
    "FunctionScalar",
    "FunctionTensor",
    "Material",
    "MaterialScalar",
    "MaterialTensor",
    "MaterialComposite",
    "MaterialHyperelastic",
    "MaterialHyperelasticPlaneStrain",
    "MaterialHyperelasticPlaneStressIncompressible",
    "MaterialHyperelasticPlaneStressLinearElastic",
    "MaterialTensorGeneral",
    "ThreeFieldVariation",
    "ThreeFieldVariationPlaneStrain",
    "TwoFieldVariation",
    "TwoFieldVariationPlaneStrain",
    "Variable",
]
