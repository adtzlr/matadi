import numpy as np

from matadi import MaterialTensorGeneral
from matadi.models import NeoHookeOgdenRoxburgh, Morph, neo_hooke, volumetric
from matadi.math import det, gradient


def fun(x, C10=0.5, bulk=5000):

    F = x[0]
    J = det(F)

    W = neo_hooke(F, C10)
    U = volumetric(J, bulk)

    return gradient(W, F) + gradient(U, F), x[-1]


def test_u_templates():

    Custom = MaterialTensorGeneral(fun, nstatevars=1)

    # Material as a function of `F` and `p`
    # with additional state variables `z`
    for M in [Custom]:

        FF = (np.random.rand(3, 3, 8, 100) - 0.5) / 2
        zz = np.random.rand(*M.x[-1].shape, 8, 100)

        for a in range(3):
            FF[a, a] += 1

        P = M.function([FF, zz])  # stress, constraint, statevars_new
        A = M.gradient([FF, zz])

        assert len(P) == 2
        assert len(A) == 1


def test_up_templates():

    # Material as a function of `F` and `p`
    # with additional state variables `z`
    for M in [NeoHookeOgdenRoxburgh(), Morph()]:

        FF = (np.random.rand(3, 3, 8, 100) - 0.5) / 2
        pp = np.random.rand(1, 8, 100)
        zz = np.random.rand(*M.x[-1].shape, 8, 100)

        for a in range(3):
            FF[a, a] += 1

        P = M.function([FF, pp, zz])  # stress, constraint, statevars_new
        A = M.gradient([FF, pp, zz])

        assert len(P) == 3
        assert len(A) == 3


if __name__ == "__main__":
    test_u_templates()
    test_up_templates()
