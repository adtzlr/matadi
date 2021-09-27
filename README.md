# matADi
Material Definition with Automatic Differentiation (AD)

[![PyPI version shields.io](https://img.shields.io/pypi/v/matadi.svg)](https://pypi.python.org/pypi/matadi/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/matadi/branch/main/graph/badge.svg?token=2EY2U4ZL35)](https://codecov.io/gh/adtzlr/matadi) [![DOI](https://zenodo.org/badge/408564756.svg)](https://zenodo.org/badge/latestdoi/408564756) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![GitHub Repo stars](https://img.shields.io/github/stars/adtzlr/matadi?logo=github) ![PyPI - Downloads](https://img.shields.io/pypi/dm/matadi)

matADi is a simple Python module which acts as a wrapper on top of [casADi](https://web.casadi.org/) [1] for easy definitions of hyperelastic strain energy functions. Gradients (stresses) and hessians (elasticity tensors) are carried out by casADi's powerful and fast **Automatic Differentiation (AD)** capabilities. It is designed to handle inputs with trailing axes which is especially useful for the application in Python-based finite element modules like [scikit-fem](https://scikit-fem.readthedocs.io/en/latest/) or [FElupe](https://adtzlr.github.io/felupe/). Mixed-field formulations are supported as well as single-field formulations.

## Installation
Install `matADi` from PyPI via pip.

```shell
pip install matadi
```

## Usage
First, a symbolic variable on which our strain energy function will be based on has to be created.

**Note**: *A variable of matADi is an instance of a symbolic variable of casADi (`casadi.SX.sym`). All `matadi.math` functions are simple links to (symbolic) casADi-functions.*

```python
from matadi import Variable, Material
from matadi.math import det, transpose, trace

F = Variable("F", 3, 3)
```

Next, take your favorite paper on hyperelasticity or be creative and define your own strain energy density function as a function of some variables `x` (where `x` is always a **list** of variables).

```python
def neohooke(x, mu=1.0, bulk=200.0):
    """Strain energy density function of a nearly-incompressible 
    Neo-Hookean isotropic hyperelastic material formulation."""

    F = x[0]
    
    J = det(F)
    C = transpose(F) @ F
    I1_iso = J ** (-2 / 3) * trace(C)

    return mu * (I1_iso - 3) + bulk * (J - 1) ** 2 / 2
```

With this simple Python function at hand, we create an instance of a **Material**, which allows extra `args` and `kwargs` to be passed to our strain energy function. This instance now enables the evaluation of both **gradient** (stress) and **hessian** (elasticity) via methods based on automatic differentiation - optionally also on input data containing trailing axes. If necessary, the strain energy density function itself will be evaluated on input data with optional trailing axes by the **function** method.

```python
Mat = Material(
    x=[F],
    fun=neohooke,
    kwargs={"mu": 1.0, "bulk": 10.0},
)

# init some random deformation gradients
defgrad = np.random.rand(3, 3, 5, 100) - 0.5

for a in range(3):
    defgrad[a, a] += 1.0

W = Mat.function([defgrad])[0]
P = Mat.gradient([defgrad])[0]
A = Mat.hessian([defgrad])[0]
```

## Template classes for hyperelasticity
matADi provides several simple template classes suitable for simple hyperelastic materials. Some common isotropic hyperelastic material formulations are located in `matadi.models` (see list below). These strain energy functions have to be passed as the `fun` argument into an instance of `MaterialHyperelastic`. Usage is exactly the same as described above. To convert a hyperelastic material based on the deformation gradient into a mixed three-field formulation suitable for nearly-incompressible behavior (*displacements*, *pressure* and *volume ratio*) an instance of a `MaterialHyperelastic` class has to be passed to `ThreeFieldVariation`.

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

W_upJ = ThreeFieldVariation(NH).function([defgrad, pressure, volratio])
P_upJ = ThreeFieldVariation(NH).gradient([defgrad, pressure, volratio])
A_upJ = ThreeFieldVariation(NH).hessian([defgrad, pressure, volratio])
```

Available isotropic hyperelastic material models:
- [Neo-Hooke](https://en.wikipedia.org/wiki/Neo-Hookean_solid)
- [Mooney-Rivlin](https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid)
- [Yeoh](https://en.wikipedia.org/wiki/Yeoh_(hyperelastic_model))
- [Third-Order-Deformation (James-Green-Simpson)](https://onlinelibrary.wiley.com/doi/abs/10.1002/app.1975.070190723)
- [Ogden](https://en.wikipedia.org/wiki/Ogden_(hyperelastic_model))
- [Arruda-Boyce](https://en.wikipedia.org/wiki/Arruda%E2%80%93Boyce_model)

Available anisotropic hyperelastic material models:
- Fiber
- Fiber-family (+/- combination of single Fiber)
- [Holzapfel Gasser Ogden](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2005.0073)

Any user-defined isotropic hyperelastic strain energy density function may be passed as the `fun` argument of `MaterialHyperelastic` by using the following template:

```python
def fun(F, **kwargs):
    # user code
    return W
```

## Lab
In the `Lab` experiments on homogenous loadcases can be performed. Let's take the above neo-hookean material formulation and run **uniaxial**, **biaxial** and **planar shear** tests.

```python
from matadi import Lab

lab = Lab(NH)
data = lab.run(ux=True, bx=True, ps=True)
fig, ax = lab.plot(data)
```

![Lab experiments(Neo-Hooke)](https://raw.githubusercontent.com/adtzlr/matadi/main/docs/images/plot_lab-nh.svg)

Unstable states of deformation can be indicated as dashed lines with the stability argument `lab.plot(data, stability=True)`. This checks if 
a) the volume ratio is greater zero,
b) the slope of stress vs. stretch and
c) the sign of the resulting stretch from a small superposed force in one direction.

## Hints
Please have a look at [casADi's documentation](https://web.casadi.org/). It is very powerful but unfortunately does not support all the Python stuff you would expect. For example Python's default if-else-statements can't be used in combination with symbolic conditions (use `math.if_else(cond, if_true, if_false)` instead).

## References
[1] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, *CasADi - A software framework for nonlinear optimization and optimal control*, Math. Prog. Comp., vol. 11, no. 1, pp. 1–36, 2019, [![DOI:10.1007/s12532-018-0139-4](https://zenodo.org/badge/DOI/10.1007/s12532-018-0139-4.svg)](https://doi.org/10.1007/s12532-018-0139-4)