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
    MaterialHyperelastic,
    MaterialComposite,
)
from ._lab import Lab
