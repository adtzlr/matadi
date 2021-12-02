import numpy as np

from matadi import (
    MaterialHyperelasticPlaneStrain,
    MaterialHyperelasticPlaneStressIncompressible,
    MaterialHyperelasticPlaneStressLinearElastic,
    ThreeFieldVariationPlaneStrain,
)
from matadi.models import neo_hooke, linear_elastic


def pre():

    FF = np.random.rand(2, 2, 8, 1000)
    for a in range(2):
        FF[a, a] += 1

    return FF


def pre_mixed():

    FF = np.random.rand(2, 2, 8, 1000)
    for a in range(2):
        FF[a, a] += 1

    pp = np.random.rand(1, 1, 8, 1000)
    JJ = np.random.rand(1, 1, 8, 1000) + 1

    return FF, pp, JJ


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


def test_plane_strain_mixed():

    # data
    FF, pp, JJ = pre_mixed()

    # init Material
    W = MaterialHyperelasticPlaneStrain(
        fun=neo_hooke,
        C10=0.5,
    )

    W_upJ = ThreeFieldVariationPlaneStrain(W)

    W0 = W_upJ.function([FF, pp, JJ])
    dW = W_upJ.gradient([FF, pp, JJ])
    DW = W_upJ.hessian([FF, pp, JJ])

    assert W0[0].shape == (8, 1000)
    assert dW[0].shape == (2, 2, 8, 1000)
    assert DW[0].shape == (2, 2, 2, 2, 8, 1000)

    assert W0[0].shape == (8, 1000)
    assert dW[0].shape == (2, 2, 8, 1000)
    assert DW[0].shape == (2, 2, 2, 2, 8, 1000)


def test_plane_stress_incompr():

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


def test_plane_stress_linear():

    # data
    FF = pre()

    # init Material
    W = MaterialHyperelasticPlaneStressLinearElastic(
        fun=linear_elastic,
        mu=1.0,
        lmbda=200.0,
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
    test_plane_strain_mixed()
    test_plane_stress_incompr()
    test_plane_stress_linear()
