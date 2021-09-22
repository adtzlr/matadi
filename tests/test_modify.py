import numpy as np

from matadi import Variable, Material
from matadi.math import det, transpose, trace


def test_modify():

    # variables
    F = Variable("F", 3, 3)

    def neohooke(x, mu=1.0, bulk=200.0):
        """Strain energy density function of nearly-incompressible
        Neo-Hookean isotropic hyperelastic material formulation."""

        F = x[0]

        J = det(F)
        C = transpose(F) @ F
        I1_iso = J ** (-2 / 3) * trace(C)

        return mu * (I1_iso - 3) + bulk * (J - 1) ** 2 / 2

    # data
    FF = np.random.rand(3, 3, 5, 100)
    for a in range(3):
        FF[a, a] += 1

    # init Material
    W = Material(
        x=[F],
        fun=neohooke,
        kwargs={"mu": 1.0, "bulk": 10.0},
    )

    dW = W.gradient([FF])
    DW = W.hessian([FF])

    dW2 = W.gradient([FF], modify=[True])
    DW2 = W.hessian([FF], modify=[True])

    # dW and DW are always lists...
    assert dW[0].shape == (3, 3, 5, 100)
    assert DW[0].shape == (3, 3, 3, 3, 5, 100)

    assert dW2[0].shape == (3, 3, 5, 100)
    assert DW2[0].shape == (3, 3, 3, 3, 5, 100)


if __name__ == "__main__":
    test_modify()
