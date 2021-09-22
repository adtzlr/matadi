import numpy as np

from matadi import Variable, Material
from matadi.math import transpose, eigvals, sum1, trace


def test_eigvals():

    # variables
    F = Variable("F", 3, 3)

    def fun(x):
        F = x[0]
        return (sum1(eigvals(transpose(F) @ F)) - 3) / 2

    # data
    np.random.seed(2345537)
    FF = np.random.rand(3, 3, 5, 100)
    for a in range(3):
        FF[a, a] += 1

    FF[:, :, 0, 1] = np.diag([1.2, 0.7, 1.2])

    # init Material
    W = Material(x=[F], fun=fun)

    WW = W.function([FF], modify=[True], eps=1e-6)
    dW = W.gradient([FF], modify=[True], eps=1e-6)
    DW = W.hessian([FF], modify=[True], eps=1e-6)

    Eye = np.eye(3)
    Eye4 = np.einsum("ij,kl->ikjl", Eye, Eye)

    # check function
    f = FF[:, :, 0, 0]
    c = f.T @ f
    assert np.isclose(WW[0][0, 0], (np.linalg.eigvals(c).sum() - 3) / 2)

    # check gradient
    assert np.allclose(dW[0][:, :, 0, 0], FF[:, :, 0, 0])
    assert np.allclose(dW[0][:, :, 0, 1], FF[:, :, 0, 1])

    # check hessian
    assert np.allclose(DW[0][:, :, :, :, 0, 0], Eye4)
    assert np.allclose(DW[0][:, :, :, :, 0, 1], Eye4)


def test_eigvals_single():

    # variables
    F = Variable("F", 3, 3)

    def fun(x):
        F = x[0]
        return (sum1(eigvals(transpose(F) @ F))[0, 0] - 3) / 2

    # data
    FF = np.diag([1.2, 0.7, 1.2])

    # init Material
    W = Material(x=[F], fun=fun)
    
    WW = W.function([FF])
    WW = W.function([FF], modify=[True], eps=1e-6)
    dW = W.gradient([FF], modify=[True], eps=1e-6)
    DW = W.hessian([FF], modify=[True], eps=1e-6)

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


if __name__ == "__main__":
    test_eigvals()
    test_eigvals_single()
