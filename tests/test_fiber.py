import numpy as np

from matadi import Variable, MaterialHyperelastic
from matadi.models import fiber, fiber_family
from matadi.math import det, transpose, trace, invariants, sqrt


def test_fiber():

    # data
    FF = np.zeros((3, 3, 2))
    for a in range(3):
        FF[a, a] += 1

    for model in [fiber, fiber_family]:

        # init Material
        M = MaterialHyperelastic(model, E=1, angle=30, axis=2)

        W0 = M.function([FF])
        dW = M.gradient([FF])
        DW = M.hessian([FF])

        assert W0[0].shape == (2,)
        assert dW[0].shape == (3, 3, 2,)
        assert DW[0].shape == (3, 3, 3, 3, 2,)

        print(dW[0])


if __name__ == "__main__":
    test_fiber()
