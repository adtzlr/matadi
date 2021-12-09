import numpy as np

from matadi import Variable, MaterialHyperelastic
from matadi.models import fiber, fiber_family, holzapfel_gasser_ogden
from matadi.math import det, transpose, trace, invariants, sqrt


def test_fiber():

    # data
    FF = np.zeros((3, 3, 2))
    for a in range(3):
        FF[a, a] += 1

    for model in [fiber, fiber_family]:

        # init Material without bulk
        M = MaterialHyperelastic(model, E=1, angle=30, axis=2, k=0)

        W0 = M.function([FF])
        assert W0[0].shape == (2,)

        # init Material
        M = MaterialHyperelastic(model, E=1, angle=30, axis=2, bulk=5000)

        W0 = M.function([FF])
        dW = M.gradient([FF])
        DW = M.hessian([FF])

        assert W0[0].shape == (2,)
        assert dW[0].shape == (
            3,
            3,
            2,
        )
        assert DW[0].shape == (
            3,
            3,
            3,
            3,
            2,
        )


def test_hgo():

    # data
    FF = np.zeros((3, 3, 2))
    for a in range(3):
        FF[a, a] += 1

    for model in [holzapfel_gasser_ogden]:

        # init Material without bulk
        M = MaterialHyperelastic(
            model, c=0.0764, k1=996.6, k2=524.6, kappa=0.2, angle=49.98, axis=2
        )

        # init Material
        M = MaterialHyperelastic(
            model,
            c=0.0764,
            k1=996.6,
            k2=524.6,
            kappa=0.2,
            angle=49.98,
            axis=2,
            bulk=5000,
        )

        W0 = M.function([FF])
        dW = M.gradient([FF])
        DW = M.hessian([FF])

        assert W0[0].shape == (2,)
        assert dW[0].shape == (
            3,
            3,
            2,
        )
        assert DW[0].shape == (
            3,
            3,
            3,
            3,
            2,
        )


if __name__ == "__main__":
    test_fiber()
    test_hgo()
