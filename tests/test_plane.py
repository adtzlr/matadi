import numpy as np

from matadi import (
    MaterialHyperelasticPlaneStrain,
    MaterialHyperelasticPlaneStressIncompressible,
)
from matadi.models import neo_hooke


def pre():

    FF = np.random.rand(2, 2, 8, 1000)
    for a in range(2):
        FF[a, a] += 1

    return FF


def test_plane_strain():

    # data
    FF = pre()

    # init Material
    W = MaterialHyperelasticPlaneStrain(
        fun=neo_hooke,
        C10=0.5,
    )

    W0 = W.function([FF])
    dW = W.gradient([FF])
    DW = W.hessian([FF])

    assert W0[0].shape == (8, 1000)
    assert dW[0].shape == (2, 2, 8, 1000)
    assert DW[0].shape == (2, 2, 2, 2, 8, 1000)

    assert W0[0].shape == (8, 1000)
    assert dW[0].shape == (2, 2, 8, 1000)
    assert DW[0].shape == (2, 2, 2, 2, 8, 1000)


def test_plane_stress():

    # data
    FF = pre()

    # init Material
    W = MaterialHyperelasticPlaneStressIncompressible(
        fun=neo_hooke,
        C10=0.5,
    )

    W0 = W.function([FF])
    dW = W.gradient([FF])
    DW = W.hessian([FF])

    assert W0[0].shape == (8, 1000)
    assert dW[0].shape == (2, 2, 8, 1000)
    assert DW[0].shape == (2, 2, 2, 2, 8, 1000)

    assert W0[0].shape == (8, 1000)
    assert dW[0].shape == (2, 2, 8, 1000)
    assert DW[0].shape == (2, 2, 2, 2, 8, 1000)


if __name__ == "__main__":
    test_plane_strain()
    test_plane_stress()