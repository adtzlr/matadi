import numpy as np

from matadi import Variable, Material, MaterialTensor
from matadi.math import det, transpose, trace, invariants, sqrt, dev


def neohooke(x, mu=1.0, bulk=200.0):
    """Strain energy density function of nearly-incompressible
    Neo-Hookean isotropic hyperelastic material formulation."""

    F = x[0]
    C = transpose(F) @ F

    I1, I2, I3 = invariants(C)
    J = sqrt(I3)

    I1_iso = I3 ** (-1 / 3) * trace(C)

    return mu * (I1_iso - 3) + bulk * (J - 1) ** 2 / 2


def test_simple():

    # variables
    F = Variable("F", 3, 3)

    # data
    FF = np.random.rand(3, 3, 8, 1000)
    for a in range(3):
        FF[a, a] += 1

    # init Material
    W = Material(x=[F], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0})

    W0 = W.function([FF])
    dW = W.gradient([FF])
    DW = W.hessian([FF])

    # dW and DW are always lists...
    assert W0[0].shape == (1, 1, 8, 1000)
    assert dW[0].shape == (3, 3, 8, 1000)
    assert DW[0].shape == (3, 3, 3, 3, 8, 1000)

    # check output of parallel evaluation
    W0 = W.function([FF], threads=2)
    dW = W.gradient([FF], threads=2)
    DW = W.hessian([FF], threads=2)

    assert W0[0].shape == (1, 1, 8, 1000)
    assert dW[0].shape == (3, 3, 8, 1000)
    assert DW[0].shape == (3, 3, 3, 3, 8, 1000)

    # test single value input
    W00 = W.function([FF[:, :, 0, 0]])

    assert W00[0].shape == (1, 1)

    # init Material with compress
    W = Material(x=[F], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0}, compress=True)
    W0 = W.function([FF])

    assert W0[0].shape == (8, 1000)

    # test single value input
    W00 = W.function([FF[:, :, 0, 0]])

    assert W00[0].shape == (1,)


def test_tensor():

    # variables
    F = Variable("F", 3, 3)
    p = Variable("p", 1, 1)
    z = Variable("z", 5, 16)

    # data
    FF = np.random.rand(3, 3, 8, 1000)
    pp = np.random.rand(8, 1000)
    zz = np.random.rand(5, 16, 8, 1000)

    for a in range(3):
        FF[a, a] += 1

    # init Material
    for fun in [lambda x: dev(x[0]), lambda x: x]:

        W = MaterialTensor(x=[F], fun=fun)

        W0 = W.function([FF])
        dW = W.gradient([FF])
        DW = W.jacobian([FF])

        # dW and DW are always lists...
        assert W0[0].shape == (3, 3, 8, 1000)
        assert dW[0].shape == (3, 3, 3, 3, 8, 1000)
        assert DW[0].shape == (3, 3, 3, 3, 8, 1000)

        # check output of parallel evaluation
        W0 = W.function([FF], threads=2)
        dW = W.gradient([FF], threads=2)
        DW = W.jacobian([FF], threads=2)

        assert W0[0].shape == (3, 3, 8, 1000)
        assert dW[0].shape == (3, 3, 3, 3, 8, 1000)

    # init Material
    pp = np.random.rand(8, 1000)

    W = MaterialTensor(x=[p], fun=lambda x: x[0], compress=True)
    W0 = W.function([pp], threads=2)

    assert W0[0].shape == (8, 1000)

    pp = np.random.rand(1, 1, 8, 1000)

    W = MaterialTensor(x=[p], fun=lambda x: x[0])
    W0 = W.function([pp])

    assert W0[0].shape == (1, 1, 8, 1000)

    # init mixed Material with upper triangle gradient
    W = MaterialTensor(x=[F, p], fun=lambda x: x, triu=True)
    P = W.function([FF, pp])
    A = W.gradient([FF, pp])

    assert len(P) == 2
    assert len(A) == 3

    # init mixed Material with full gradient
    W = MaterialTensor(x=[F, p], fun=lambda x: x, triu=False)
    P = W.function([FF, pp])
    A = W.gradient([FF, pp])

    assert len(P) == 2
    assert len(A) == 4

    # init mixed Material with upper triangle gradient and state variables
    W = MaterialTensor(x=[F, p, z], fun=lambda x: [*x, z], triu=True, statevars=1)
    P = W.function([FF, pp, zz])
    A = W.gradient([FF, pp, zz])

    assert len(P) == 4
    assert len(A) == 3


if __name__ == "__main__":
    test_simple()
    test_tensor()
