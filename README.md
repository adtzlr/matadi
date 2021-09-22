# matADi
Material Definition with Automatic Differentiation (AD)

[![PyPI version shields.io](https://img.shields.io/pypi/v/matadi.svg)](https://pypi.python.org/pypi/matadi/) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) ![Made with love in Graz (Austria)](https://img.shields.io/badge/Made%20with%20%E2%9D%A4%EF%B8%8F%20in-Graz%20(Austria)-0c674a) [![codecov](https://codecov.io/gh/adtzlr/matadi/branch/main/graph/badge.svg?token=2EY2U4ZL35)](https://codecov.io/gh/adtzlr/matadi) [![DOI](https://zenodo.org/badge/408564756.svg)](https://zenodo.org/badge/latestdoi/408564756) ![Codestyle black](https://img.shields.io/badge/code%20style-black-black) ![GitHub Repo stars](https://img.shields.io/github/stars/adtzlr/matadi?logo=github) ![PyPI - Downloads](https://img.shields.io/pypi/dm/matadi)

matADi is a simple Python module which acts as a wrapper on top of [casADi](https://web.casadi.org/) for easy definitions of hyperelastic strain energy functions. Gradients (stresses) and hessians (elasticity tensors) are carried out by casADi's powerful and fast **Automatic Differentiation (AD)** capabilities. It is designed to handle inputs with trailing axes which is especially useful for the application in Python-based finite element modules like [scikit-fem](https://scikit-fem.readthedocs.io/en/latest/) or [FElupe](https://adtzlr.github.io/felupe/). Mixed-field formulations are supported as well as single-field formulations.

## Installation
Install `matADi` from PyPI via pip.

```shell
pip install matadi
```

## Usage
First, we have to define a symbolic variable on which our strain energy function will be based on.

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

With this simple Python function we create an instance of a **Material**, which allows extra `args` and `kwargs` to be passed to our strain energy function. This instance now enables the evaluation of both **gradient** (stress) and **hessian** (elasticity) via methods based on automatic differentiation - optionally also on input data containing trailing axes. If necessary, the strain energy density function itself will be evaluated on input data with optional trailing axes by the **function** method.

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
matADi provides several template classes for hyperelastic materials. Some common isotropic hyperelastic material formulations are located in `matadi.models`. These strain energy functions have to be passed into an instance of `MaterialHyperelastic`. Usage is exactly the same as described above. To convert a hyperelastic material based on the deformation gradient into a mixed three-field formulation suitable for nearly-incompressible behavior (*displacements*, *pressure* and *volume ratio*) an instance of a `MaterialHyperelastic` class has to be passed to the `ThreeFieldVariation` class.

```python

from matadi import MaterialHyperelastic, ThreeFieldVariation
from matadi.models import neo_hooke

# init some random data
pressure = np.random.rand(5, 100)
volratio = np.random.rand(5, 100) / 10 + 1

NH = MaterialHyperelastic(fun=neo_hooke, C10=0.5, bulk=20.0)

W = NH.function([defgrad])[0]
P = NH.gradient([defgrad])[0]
A = NH.hessian([defgrad])[0]

W_upJ = ThreeFieldVariation(NH).function([defgrad, pressure, volratio])
P_upJ = ThreeFieldVariation(NH).gradient([defgrad, pressure, volratio])
A_upJ = ThreeFieldVariation(NH).hessian([defgrad, pressure, volratio])
```

Available isotropic hyperelastic models:
- [x] Neo-Hooke
- [x] Mooney-Rivlin
- [x] Yeoh
- [x] Third-Order-Deformation
- [x] Ogden
- [x] Arruda-Boyce

Any user-defined isotropic hyperelastic strain energy density function may be passed as the `fun` argument of an instance of `MaterialHyperelastic` by using the following template:

```python
def fun(F, **kwargs):
    # user code
    return W
```

## Hints
Please have a look at [casADi's documentation](https://web.casadi.org/). It is very powerful but unfortunately does not support all the Python stuff you would expect. For example Python's default if-statements can't be used in combination with a symbolic boolean operation. If you use `eigvals` to symbolically calculate eigenvalues and their corresponding gradients please call gradient and hessian methods with `Mat.gradient([defgrad], modify=True, eps=1e-5)` to avoid the gradient to be filled with NaN's. This is because the gradient of the implemented eigenvalue calculation is not defined for the case of repeated equal eigenvalues. The `modify` argument adds a small number `eps=1e-5` to the diagonal entries of the input data.