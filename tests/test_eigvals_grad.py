import numpy as np

from matadi import Variable, Material, MaterialTensor
from matadi.math import transpose, eigvals, sum1, trace, cof, inv, det, SX


def fun(x):
    F = x[0]
    return (sum1(eigvals(transpose(F) @ F))[0, 0] - 3) / 2


def test_eigvals():

    # variables
    F = Variable("F", 3, 3)

    # data
    np.random.seed(2345537)
    FF = np.random.rand(3, 3, 5, 100)
    for a in range(3):
        FF[a, a] += 1

    # input with repeated equal eigenvalues
    FF[:, :, 0, 1] = np.diag(2.4 * np.ones(3))
    FF[:, :, 0, 2] = np.diag([2.4, 1.2, 2.4])

    # init Material
    W = Material(x=[F], fun=fun)

    WW = W.function([FF])
    dW = W.gradient([FF])
    DW = W.hessian([FF])

    Eye = np.eye(3)
    Eye4 = np.einsum("ij,kl->ikjl", Eye, Eye)

    # check function
    f = FF[:, :, 0, 0]
    c = f.T @ f
    assert np.isclose(WW[0][0, 0], (np.linalg.eigvals(c).sum() - 3) / 2)

    # check gradient
    assert np.allclose(dW[0][:, :, 0, 0], FF[:, :, 0, 0])
    assert np.allclose(dW[0][:, :, 0, 1], FF[:, :, 0, 1])
    assert np.allclose(dW[0][:, :, 0, 2], FF[:, :, 0, 2])

    # check hessian
    assert np.allclose(DW[0][:, :, :, :, 0, 0], Eye4)
    assert np.allclose(DW[0][:, :, :, :, 0, 1], Eye4)
    assert np.allclose(DW[0][:, :, :, :, 0, 2], Eye4)


def test_eigvals_single():

    # variables
    F = Variable("F", 3, 3)

    # data
    FF = np.diag(2.4 * np.ones(3))

    # init Material
    W = Material(x=[F], fun=fun)

    WW = W.function([FF])
    WW = W.function([FF])
    dW = W.gradient([FF])
    DW = W.hessian([FF])

    Eye = np.eye(3)
    Eye4 = np.einsum("ij,kl->ikjl", Eye, Eye)

    # check function
    f = FF
    c = f.T @ f
    assert np.isclose(WW[0], (np.linalg.eigvals(c).sum() - 3) / 2)

    # check gradient
    assert np.allclose(dW[0], FF)

    # check hessian
    assert np.allclose(DW[0], Eye4)


def test_cof():

    # variables
    F = Variable("F", 3, 3)

    # data
    FF = np.diag(2.4 * np.ones(3)).reshape(3, 3, 1, 1)

    # fun
    def g(x):
        F = x[0]
        return cof(F)

    # init Material
    W = MaterialTensor(x=[F], fun=g)
    W = MaterialTensor(x=[F], fun=lambda x: x[0])

    Eye = np.eye(3)
    Eye4 = np.einsum("ij,kl->ikjl", Eye, Eye)

    WW = W.function([FF])
    dW = W.gradient([FF])
    DW = W.jacobian([FF])

    assert np.allclose(dW, DW)
    assert np.allclose(dW[0][:, :, :, :, 0, 0], Eye4)


if __name__ == "__main__":
    test_eigvals()
    test_eigvals_single()
    test_cof()
