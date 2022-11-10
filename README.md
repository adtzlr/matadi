# matADi
Material Definition with Automatic Differentiation (AD)

![matADi](https://raw.githubusercontent.com/adtzlr/matadi/main/docs/logo/matadi-logo.svg)

[![PyPI version shields.io](https://img.shields.io/pypi/v/matadi.svg)](https://pypi.python.org/pypi/matadi/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/matadi/branch/main/graph/badge.svg?token=2EY2U4ZL35)](https://codecov.io/gh/adtzlr/matadi) [![DOI](https://zenodo.org/badge/408564756.svg)](https://zenodo.org/badge/latestdoi/408564756) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![GitHub Repo stars](https://img.shields.io/github/stars/adtzlr/matadi?logo=github) ![PyPI - Downloads](https://img.shields.io/pypi/dm/matadi)

matADi is a Python module for the definition of a constitutive hyperelastic material formulation by a strain energy density function. Both [gradient](https://en.wikipedia.org/wiki/Gradient) (stress) and [hessian](https://en.wikipedia.org/wiki/Hessian_matrix) (elasticity tensor) are carried out by [**Automatic Differentiation (AD)**](https://en.wikipedia.org/wiki/Automatic_differentiation) using [casADi](https://web.casadi.org/) [[1](https://doi.org/10.1007/s12532-018-0139-4)] as a backend. A high-level interface for hyperelastic materials based on the deformation gradient exists. Several isotropic hyperelastic material formulations like the Neo-Hookean or the Ogden model are included in the model library. Gradient and hessian methods allow input data with trailing axes which is especially useful for Python-based finite element modules, e.g. [scikit-fem](https://scikit-fem.readthedocs.io/en/latest/) or [FElupe](https://github.com/adtzlr/felupe). Mixed-field formulations suitable for nearly-incompressible material behavior are supported as well as single-field formulations based on the deformation gradient. A numerical lab is included to run, plot and analyze homogenous uniaxial, equi-biaxial, planar shear and simple shear load cases.

## Installation
Install `matADi` from PyPI via pip.

```shell
pip install matadi
```

## Usage
First, a symbolic variable on which our strain energy function will be based on has to be created.

**Note**: *A variable of matADi is an instance of a symbolic variable of casADi (`casadi.SX.sym`). Most (but not all) of `matadi.math` functions are simple links to (symbolic) casADi-functions.*

```python
from matadi import Variable, Material
from matadi.math import det, transpose, trace

F = Variable("F", 3, 3)
```

Next, take your favorite paper on hyperelasticity or be creative and define your own strain energy density function as a function of some variables `x` (where `x` is always a **list** of variables).

```python
def neohooke(x, C10=0.5, bulk=200.0):
    """Strain energy density function of a nearly-incompressible 
    Neo-Hookean isotropic hyperelastic material formulation."""

    F = x[0]
    
    J = det(F)
    C = transpose(F) @ F

    return C10 * (J ** (-2 / 3) * trace(C) - 3) + bulk * (J - 1) ** 2 / 2
```

With this simple Python function at hand, we create an instance of a **Material**, which allows extra `args` and `kwargs` to be passed to our strain energy function. This instance now enables the evaluation of both **gradient** (stress) and **hessian** (elasticity) via methods based on automatic differentiation - optionally also on input data containing trailing axes. If necessary, the strain energy density function itself will be evaluated on input data with optional trailing axes by the **function** method.

```python
Mat = Material(
    x=[F],
    fun=neohooke,
    kwargs={"C10": 0.5, "bulk": 10.0},
)

# init some random deformation gradients
import numpy as np

defgrad = np.random.rand(3, 3, 5, 100) - 0.5

for a in range(3):
    defgrad[a, a] += 1.0

W = Mat.function([defgrad])[0]
P = Mat.gradient([defgrad])[0]
A = Mat.hessian([defgrad])[0]
```

In a similar way, gradient-vector-products and hessian-vector-products are accessible via **gradient_vector_product** and **hessian_vector_product** methods, respectively.

```python
v = np.random.rand(3, 3, 5, 100) - 0.5
u = np.random.rand(3, 3, 5, 100) - 0.5

W   = Mat.function([defgrad])[0]
dW  = Mat.gradient_vector_product([defgrad], [v])[0]
DdW = Mat.hessian_vector_product([defgrad], [v], [u])[0]
```

## Template classes for hyperelasticity
matADi provides several template classes suitable for hyperelastic materials. Some isotropic hyperelastic material formulations are located in `matadi.models` (see list below). These strain energy functions have to be passed as the `fun` argument into an instance of `MaterialHyperelastic`. Usage is exactly the same as described above. To convert a hyperelastic material based on the deformation gradient into a mixed three-field formulation suitable for nearly-incompressible behavior (*displacements*, *pressure* and *volume ratio*) an instance of a `MaterialHyperelastic` class has to be passed to `ThreeFieldVariation`. For *plane strain* or *plane stress* use `MaterialHyperelasticPlaneStrain`, `MaterialHyperelasticPlaneStressIncompressible` or `MaterialHyperelasticPlaneStressLinearElastic` instead of `MaterialHyperelastic`. For plane strain *displacements*, *pressure* and *volume ratio* mixed-field formulations use `ThreeFieldVariationPlaneStrain`. For a two-field formulation (*displacements*, *pressure*) use `TwoFieldVariation` on top of `MaterialHyperelastic`.

```python

from matadi import MaterialHyperelastic, ThreeFieldVariation
from matadi.models import neo_hooke

# init some random data
pressure = np.random.rand(5, 100)
volratio = np.random.rand(5, 100) / 10 + 1

kwargs = {"C10": 0.5, "bulk": 20.0}

NH = MaterialHyperelastic(fun=neo_hooke, **kwargs)

W = NH.function([defgrad])[0]
P = NH.gradient([defgrad])[0]
A = NH.hessian([defgrad])[0]

NH_upJ = ThreeFieldVariation(NH)

W_upJ = NH_upJ.function([defgrad, pressure, volratio])
P_upJ = NH_upJ.gradient([defgrad, pressure, volratio])
A_upJ = NH_upJ.hessian([defgrad, pressure, volratio])
```

The output of `NH_upJ.gradient([defgrad, pressure, volratio])` is a list with gradients of the functional as `[dWdF, dWdp, dWdJ]`. Hessian entries are provided as a list of upper triangle entries, e.g. `NH_upJ.hessian([defgrad, pressure, volratio])` returns `[d2WdFdF, d2WdFdp, d2WdFdJ, d2Wdpdp, d2WdpdJ, d2WdJdJ]`.

Available [isotropic hyperelastic material models](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py):
- [Linear Elastic](https://en.wikipedia.org/wiki/Linear_elasticity) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L5-L7))
- [Saint Venant Kirchhoff](https://en.wikipedia.org/wiki/Hyperelastic_material#Saint_Venant%E2%80%93Kirchhoff_model) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L10-L14))
- [Neo-Hooke](https://en.wikipedia.org/wiki/Neo-Hookean_solid) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L17-L21))
- [Mooney-Rivlin](https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L24-L29))
- [Yeoh](https://en.wikipedia.org/wiki/Yeoh_(hyperelastic_model)) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L32-L37))
- [Third-Order-Deformation (James-Green-Simpson)](https://onlinelibrary.wiley.com/doi/abs/10.1002/app.1975.070190723) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L40-L51))
- [Ogden](https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model)) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L54-L64))
- [Arruda-Boyce](https://en.wikipedia.org/wiki/Arruda%E2%80%93Boyce_model) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L67-L80))
- [Extended-Tube](https://meridian.allenpress.com/rct/article-abstract/72/4/602/92819/An-Extended-Tube-Model-for-Rubber-Elasticity?redirectedFrom=fulltext) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L83-L91))
- [Van-der-Waals (Kilian)](https://doi.org/10.1016/0032-3861(81)90200-7) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_isotropic.py#L94-L103))

Available [anisotropic hyperelastic material models](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_anisotropic.py):
- Fiber ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_anisotropic.py#L17-L35))
- Fiber-family (+/- combination of single Fiber) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_anisotropic.py#L38-L45))
- [Holzapfel Gasser Ogden](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2005.0073) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_hyperelasticity_anisotropic.py#L48-L77))

Available [micro-sphere hyperelastic frameworks](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere) (Miehe, Göktepe, Lulei) [[2](https://doi.org/10.1016/j.jmps.2004.03.011)]:
- [affine stretch](https://doi.org/10.1016/j.jmps.2004.03.011) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere/affine/_models.py#L6-L16))
- [affine tube](https://doi.org/10.1016/j.jmps.2004.03.011) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere/affine/_models.py#L19-L30))
- [non-affine stretch](https://doi.org/10.1016/j.jmps.2004.03.011) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere/nonaffine/_models.py#L7-L17))
- [non-affine tube](https://doi.org/10.1016/j.jmps.2004.03.011) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere/nonaffine/_models.py#L20-L32))

Available [micro-sphere hyperelastic material models](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere) (Miehe, Göktepe, Lulei) [[2](https://doi.org/10.1016/j.jmps.2004.03.011)]:
- [Miehe Göktepe Lulei](https://doi.org/10.1016/j.jmps.2004.03.011) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/microsphere/nonaffine/_models.py#L35-L49))

Any user-defined isotropic hyperelastic strain energy density function may be passed as the `fun` argument of `MaterialHyperelastic` by using the following template:

```python
def fun(F, **kwargs):
    # user code
    return W
```

In order to apply the above material model only on the isochoric part of the deformation gradient [[3](https://doi.org/10.1016/0045-7825(85)90033-7)], use the decorator [`@isochoric_volumetric_split`](https://github.com/adtzlr/matadi/blob/main/matadi/models/_helpers.py#L7-L31). If the keyword `bulk` is passed, an additional [volumetric strain energy function](https://github.com/adtzlr/matadi/blob/main/matadi/models/_helpers.py#L34-L35) is added to the base material formulation.

```python
from matadi.models import isochoric_volumetric_split
from matadi.math import trace, transpose

@isochoric_volumetric_split
def nh(F, C10=0.5):
    # user code of strain energy function, e.g. neo-hooke
    return C10 * (trace(transpose(F) @ F) - 3)

NH = MaterialHyperelastic(nh, C10=0.5, bulk=200.0)
```

## Lab
In matADi's `Lab` :lab_coat: numeric experiments on homogenous loadcases on compressible or nearly-incompressible material formulations are performed. For incompressible materials use `LabIncompressible` instead. Let's take the non-affine micro-sphere material model suitable for rubber elasticity with parameters from [[2](https://doi.org/10.1016/j.jmps.2004.03.011), Fig. 19] and run **uniaxial**, **biaxial** and **planar shear** tests.

```python
from matadi import Lab, MaterialHyperelastic
from matadi.models import miehe_goektepe_lulei

mat = MaterialHyperelastic(
    miehe_goektepe_lulei, 
    mu=0.1475, 
    N=3.273, 
    p=9.31, 
    U=9.94, 
    q=0.567, 
    bulk=5000.0,
)

lab = Lab(mat)
data = lab.run(
    ux=True, 
    bx=True, 
    ps=True, 
    shear=True, 
    stretch_min=1.0, 
    stretch_max=2.0, 
    shear_max=1.0,
)
fig, ax = lab.plot(data, stability=True)
fig2, ax2 = lab.plot_shear(data)
```

![Lab experiments(Microsphere)](https://raw.githubusercontent.com/adtzlr/matadi/main/docs/images/plot_lab-microsphere.svg)

![Lab experiments shear(Microsphere)](https://raw.githubusercontent.com/adtzlr/matadi/main/docs/images/plot_shear_lab-microsphere.svg)

Unstable states of deformation can be indicated as dashed lines with the stability argument `lab.plot(data, stability=True)`. This checks whether if 
a) the volume ratio is greater zero,
b) the monotonic increasing slope of force per undeformed area vs. stretch and
c) the sign of the resulting stretch from a small superposed force in one direction.

## Hints and usage in FEM modules
For tensor-valued material definitions use `MaterialTensor` (e.g. any stress-strain relation). Also, please have a look at [casADi's documentation](https://web.casadi.org/). It is very powerful but unfortunately does not support all the Python stuff you would expect. For example Python's default if-else-statements can't be used in combination with symbolic conditions (use `math.if_else(cond, if_true, if_false)` instead). Contrary to [casADi](https://web.casadi.org/), the gradient of the eigenvalue function is stabilized by a perturbation of the diagonal components.

### A **Material** with state variables
A generalized material model with optional state variables for the (u/p)-formulation is created by an instance of `MaterialTensor`. If the argument `triu` is set to `True` the gradient method returns only the upper triangle entries of the gradient components. If some of the input variables are internal state variables the number of these variables have to be passed to the optional argument `statevars`. While the hyperelastic material classes are defined by a strain energy function, this one is defined by the first Piola-Kirchhoff stress tensor. Internally, state variables are equal to default variables but they are excluded from gradient calculations. State variables may also be used as placeholders for additional quantities, e.g. the initial deformation gradient at the beginning of an increment or the time increment. Hence, it is a very flexible class not restricted to hyperelasticity. For consistency, the methods `gradient` and `hessian` of a tensor-based material refer to the gradient and hessian of the strain energy function.

Included [pseudo-elastic material models](https://github.com/adtzlr/matadi/blob/main/matadi/models/_pseudo_elasticity.py):
- [Ogden-Roxburgh](https://doi.org/10.1098%2Frspa.1999.0431) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_pseudo_elasticity.py#L4-L16))

Included [other material models](https://github.com/adtzlr/matadi/blob/main/matadi/models/_misc.py):
- [MORPH](https://doi.org/10.1016/S0749-6419(02)00091-8) ([code](https://github.com/adtzlr/matadi/blob/main/matadi/models/_misc.py#L19-L75))

```python
import numpy as np

from matadi.models import (
    displacement_pressure_split, neo_hooke, volumetric, ogden_roxburgh
)
from matadi import MaterialTensor, Variable
from matadi.math import det, gradient

F = Variable("F", 3, 3)
z = Variable("z", 1, 1)

@displacement_pressure_split
def fun(x, C10=0.5, bulk=5000, r=3, m=1, beta=0):
    "Strain energy function: Neo-Hooke & Ogden-Roxburgh."
    
    # split `x` into the deformation gradient and the state variable
    F, Wmaxn = x[0], x[-1]
    
    # isochoric and volumetric parts of the hyperelastic 
    # strain energy function
    W = neo_hooke(F, C10)
    U = volumetric(det(F), bulk)
    
    # pseudo-elastic softening function
    eta, Wmax = ogden_roxburgh(W, Wmaxn, r, m, beta)
    
    # first Piola-Kirchhoff stress and updated state variable
    # for a pseudo-elastic material formulation
    return eta * gradient(W, F) + gradient(U, F), Wmax

# get pressure variable from augmented function
p = fun.p

# Material as a function of `F` and `p`
# with **one** additional state variable `z`
NH = MaterialTensor(x=[F, p, z], fun=fun, triu=True, statevars=1)

defgrad = np.random.rand(3, 3, 5, 100) - 0.5
pressure = np.random.rand(1, 5, 100)
statevars = np.random.rand(1, 1, 5, 100)

for a in range(3):
    defgrad[a, a] += 1.0

dWdF, dWdp, statevars_new = NH.gradient([defgrad, pressure, statevars])
d2WdFdF, d2WdFdp, d2Wdpdp = NH.hessian([defgrad, pressure, statevars])
```

**Hint**: *The above Neo-Hooke as well as the MORPH material model formulation, both within the u/p-framework, are available as ready-to-go materials in `matadi.models` as [`NeoHookeOgdenRoxburgh()`](https://github.com/adtzlr/matadi/blob/main/matadi/models/_templates.py) and [`Morph()`](https://github.com/adtzlr/matadi/blob/main/matadi/models/_templates.py).*

**Hint**: *The state variable concept is also implemented for the `Material` class.*

Simple examples for using `matadi` with [`scikit-fem`](https://github.com/adtzlr/matadi/discussions/14#) as well as with [`felupe`](https://github.com/adtzlr/matadi/discussions/22) are shown in the Discussion section.

## References
[1] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, *CasADi - A software framework for nonlinear optimization and optimal control*, Math. Prog. Comp., vol. 11, no. 1, pp. 1–36, 2019, [![DOI:10.1007/s12532-018-0139-4](https://zenodo.org/badge/DOI/10.1007/s12532-018-0139-4.svg)](https://doi.org/10.1007/s12532-018-0139-4)

[2] C. Miehe, S. Göktepe and F. Lulei, *A micro-macro approach to rubber-like materials. Part I: the non-affine micro-sphere model of rubber elasticity*, Journal of the Mechanics and Physics of Solids, vol. 52, no. 11. Elsevier BV, pp. 2617–2660, Nov. 2004. [![DOI:10.1016/j.jmps.2004.03.011](https://zenodo.org/badge/DOI/10.1016/j.jmps.2004.03.011.svg)](https://doi.org/10.1016/j.jmps.2004.03.011)

[3] J. C. Simo, R. L. Taylor, and K. S. Pister, *Variational and projection methods for the volume constraint in finite deformation elasto-plasticity*, Computer Methods in Applied Mechanics and Engineering, vol. 51, no. 1–3, pp. 177–208, Sep. 1985, [![DOI:10.1016/0045-7825(85)90033-7](https://zenodo.org/badge/DOI/10.1016/0045-7825(85)90033-7.svg)](https://doi.org/10.1016/0045-7825(85)90033-7)

