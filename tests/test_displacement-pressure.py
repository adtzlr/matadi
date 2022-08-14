import numpy as np

from matadi import Variable, MaterialTensor
from matadi.models import (
    displacement_pressure_split,
    neo_hooke,
    ogden_roxburgh,
    volumetric,
)

from matadi.math import (
    det, gradient, dev, inv, asvoigt, astensor, vertcat, triu, zeros
)

import pytest


def test_up_state():

    # deformation gradient
    F = Variable("F", 3, 3)

    # state variables
    z = Variable("z", 1, 1)

    @displacement_pressure_split
    def fun(x, C10=0.5, bulk=5000, r=3, m=1, beta=0):
        "Strain energy function: Neo-Hooke & Ogden-Roxburgh."

        # split `x` into the deformation gradient and the state variable
        F, Wmaxn = x[0], x[-1]
        
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

        # isochoric and volumetric parts of the hyperelastic strain energy function
        W = neo_hooke(F, C10)
        U = volumetric(det(F), bulk)

        # pseudo-elastic softening
        eta, Wmax = ogden_roxburgh(W, Wmaxn, r, m, beta)

        # softened first Piola-Kirchhoff stress and updated state variable
        return eta * gradient(W, F) + gradient(U, F), Wmax

    # get pressure variable from augmented function
    p = fun.p

    # Material as a function of `F` and `p`
    # with additional state variables `z`
    M = MaterialTensor([F, p, z], fun, triu=True, statevars=1)

    FF = np.random.rand(3, 3, 8, 100)
    pp = np.random.rand(1, 8, 100)
    zz = np.random.rand(1, 1, 8, 100)

    for a in range(3):
        FF[a, a] += 1

    P = M.function([FF, pp, zz])  # stress, constraint, statevars_new
    A = M.gradient([FF, pp, zz])

    assert len(P) == 3
    assert len(A) == 3


def test_up():

    # deformation gradient
    F = Variable("F", 3, 3)

    @displacement_pressure_split
    def fun(x):
        F = x[0]
        C = F.T @ F
        return det(F) ** (-2 / 3) * dev(C) @ inv(C)

    # get pressure variable from augmented function
    p = fun.p

    # Material as a function of `F` and `p`
    M = MaterialTensor([F, p], fun, triu=True)

    FF = np.random.rand(3, 3, 8, 100)
    pp = np.random.rand(1, 8, 100)

    for a in range(3):
        FF[a, a] += 1

    P = M.function([FF, pp])  # stress, constraint
    A = M.gradient([FF, pp])

    assert len(P) == 2
    assert len(A) == 3


if __name__ == "__main__":
    test_up()
    test_up_state()
