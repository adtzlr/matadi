import numpy as np

from matadi import Variable, Material
from matadi.math import det, transpose, trace


def test_scalar():

    # variables
    F = Variable("F", 3, 3)
    p = Variable("p", 1)
    J = Variable("J", 1)

    def neohooke(x, mu=1.0, bulk=200.0):
        "Strain energy function of nearly-incompressible Neo-Hookean material."

        F, p, theta = x
        J = det(F)
        C = transpose(F) @ F
        I1 = J ** (-2 / 3) * trace(C)

        return mu * (I1 - 3) + bulk * (theta - 1) ** 2 / 2 + p * (J - theta)

    # data
    FF = np.random.rand(3, 3, 5, 100)
    pp = np.random.rand(5, 100)
    JJ = np.random.rand(5, 100)

    # functional
    W = Material(
        x=[F, p, J], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0}, compress=False
    )

    dW = W.gradient([FF, pp, JJ])
    dW2 = W.jacobian([FF, pp, JJ])
    DW = W.hessian([FF, pp, JJ])

    assert dW[0].shape == (3, 3, 5, 100)
    assert dW[1].shape == (1, 1, 5, 100)
    assert dW[2].shape == (1, 1, 5, 100)

    assert dW2[2].shape == dW2[2].shape

    assert DW[0].shape == (3, 3, 3, 3, 5, 100)
    assert DW[1].shape == (3, 3, 1, 1, 5, 100)
    assert DW[2].shape == (3, 3, 1, 1, 5, 100)
    assert DW[3].shape == (1, 1, 1, 1, 5, 100)
    assert DW[4].shape == (1, 1, 1, 1, 5, 100)
    assert DW[5].shape == (1, 1, 1, 1, 5, 100)

    # dataset 2
    FF = np.random.rand(3, 3, 5, 100)
    pp = np.random.rand(1, 5, 100)
    JJ = np.random.rand(1, 5, 100)

    # functional
    W = Material(
        x=[F, p, J], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0}, compress=False
    )

    dW = W.gradient([FF, pp, JJ])
    DW = W.hessian([FF, pp, JJ])


def test_scalar_compress():

    # variables
    F = Variable("F", 3, 3)
    p = Variable("p", 1)
    J = Variable("J", 1)

    def neohooke(x, mu=1.0, bulk=200.0):
        "Strain energy function of nearly-incompressible Neo-Hookean material."

        F, p, theta = x
        J = det(F)
        C = transpose(F) @ F
        I1 = J ** (-2 / 3) * trace(C)

        return mu * (I1 - 3) + bulk * (theta - 1) ** 2 / 2 + p * (J - theta)

    # data
    FF = np.random.rand(3, 3, 5, 100)
    pp = np.random.rand(5, 100)
    JJ = np.random.rand(5, 100)

    # functional
    W = Material(
        x=[F, p, J], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0}, compress=True
    )

    dW = W.gradient([FF, pp, JJ])
    DW = W.hessian([FF, pp, JJ])

    assert dW[0].shape == (3, 3, 5, 100)
    assert dW[1].shape == (5, 100)
    assert dW[2].shape == (5, 100)

    assert DW[0].shape == (3, 3, 3, 3, 5, 100)
    assert DW[1].shape == (3, 3, 5, 100)
    assert DW[2].shape == (3, 3, 5, 100)
    assert DW[3].shape == (5, 100)
    assert DW[4].shape == (5, 100)
    assert DW[5].shape == (5, 100)

    # dataset 2
    FF = np.random.rand(3, 3, 5, 100)
    pp = np.random.rand(1, 5, 100)
    JJ = np.random.rand(1, 5, 100)

    # functional
    W = Material(
        x=[F, p, J], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0}, compress=False
    )

    dW = W.gradient([FF, pp, JJ])
    DW = W.hessian([FF, pp, JJ])


if __name__ == "__main__":
    test_scalar()
    test_scalar_compress()
