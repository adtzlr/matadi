import numpy as np

from matadi import Variable, Material
from matadi.math import det, transpose, trace, invariants, sqrt


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
    FF = np.random.rand(3, 3, 5, 100)
    for a in range(3):
        FF[a, a] += 1

    # init Material
    W = Material(x=[F], fun=neohooke, kwargs={"mu": 1.0, "bulk": 10.0},)

    dW = W.gradient([FF])
    DW = W.hessian([FF])

    # dW and DW are always lists...
    assert dW[0].shape == (3, 3, 5, 100)
    assert DW[0].shape == (3, 3, 3, 3, 5, 100)


if __name__ == "__main__":
    test_simple()
