import numpy as np

from matadi import Variable, MaterialTensor
from matadi.models import (
    displacement_pressure_split,
    neo_hooke,
    ogden_roxburgh,
    morph,
    volumetric,
)

from matadi.math import det, gradient, dev, inv, asvoigt, astensor, vertcat, triu, zeros

import pytest


@displacement_pressure_split
def fun_nh_or(x, C10=0.5, bulk=5000, r=3, m=1, beta=0):
    "Strain energy function: Neo-Hooke & Ogden-Roxburgh."

    # split `x` into the deformation gradient and the state variable
    F, Wmaxn = x[0], x[-1]

    # isochoric and volumetric parts of the hyperelastic strain energy function
    W = neo_hooke(F, C10)
    U = volumetric(det(F), bulk)

    # pseudo-elastic softening
    eta, Wmax = ogden_roxburgh(W, Wmaxn, r, m, beta)

    # softened first Piola-Kirchhoff stress and updated state variable
    return eta * gradient(W, F) + gradient(U, F), Wmax


@displacement_pressure_split
def fun_morph(x, p1=0.035, p2=0.37, p3=0.17, p4=2.4, p5=0.01, p6=6.4, p7=5.5, p8=0.24):
    "Strain energy function: Neo-Hooke & Ogden-Roxburgh."

    # split `x` into the deformation gradient and the state variable
    F = x[0]

    P, statevars = morph(x, p1, p2, p3, p4, p5, p6, p7, p8)
    U = volumetric(det(F), 5000 * 2 * (p1 + p2))

    # first Piola-Kirchhoff stress and updated state variable
    return P + gradient(U, F), statevars


def test_up_state():

    # deformation gradient
    F = Variable("F", 3, 3)

    # state variables
    statevars = [Variable("z", 1, 1), Variable("z", 13, 1)]
    functions = [fun_nh_or, fun_morph]

    for fun, z in zip(functions, statevars):

        # get pressure variable from augmented function
        p = fun.p

        # Material as a function of `F` and `p`
        # with additional state variables `z`
        M = MaterialTensor([F, p, z], fun, triu=True, statevars=1)

        FF = (np.random.rand(3, 3, 8, 100) - 0.5) / 2
        pp = np.random.rand(1, 8, 100)
        zz = np.random.rand(*z.shape, 8, 100)

        for a in range(3):
            FF[a, a] += 1

        P = M.gradient([FF, pp, zz])  # stress, constraint, statevars_new
        A = M.hessian([FF, pp, zz])

        assert len(P) == 3
        assert len(A) == 3


def test_up_basic():

    # deformation gradient
    F = Variable("F", 3, 3)

    @displacement_pressure_split
    def fun(x):
        F = x[0]
        C = F.T @ F

        # (begin) test `asvoigt()` and `astensor()`
        C_3D = F.T @ F
        C_2D = vertcat(C_3D[0, 0], C_3D[1, 0], C_3D[0, 1], C_3D[1, 1]).reshape((2, 2))

        C_6 = asvoigt(C_3D)
        C_4 = asvoigt(C_2D)

        C_from_C_6 = astensor(C_6)
        C_2D_from_C_4 = astensor(C_4)

        assert C_3D[0, 1] == C_from_C_6[0, 1]
        assert C_2D[0, 1] == C_2D_from_C_4[0, 1]

        with pytest.raises(ValueError):
            asvoigt(C_6)

        with pytest.raises(ValueError):
            astensor(C_3D)
        # (end) test `asvoigt()` and `astensor()`

        return det(F) ** (-2 / 3) * dev(C) @ inv(C)

    # get pressure variable from augmented function
    p = fun.p

    # Material as a function of `F` and `p`
    M = MaterialTensor([F, p], fun, triu=True)

    FF = np.random.rand(3, 3, 8, 100)
    pp = np.random.rand(1, 8, 100)

    for a in range(3):
        FF[a, a] += 1

    P = M.gradient([FF, pp])  # stress, constraint
    A = M.hessian([FF, pp])

    assert len(P) == 2
    assert len(A) == 3


if __name__ == "__main__":
    test_up_basic()
    test_up_state()
