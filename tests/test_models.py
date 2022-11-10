import numpy as np

import matadi
import matadi.models as md


def get_title(model):
    return "-".join([m.title() for m in model.__name__.split("_")])


def library():
    "Library with models and parameters."

    q = md.microsphere.quadrature.BazantOh(n=21)

    database = {
        md.saint_venant_kirchhoff: {"mu": 1.0, "lmbda": 20.0},
        md.neo_hooke: {"C10": 0.5},
        md.mooney_rivlin: {"C10": 0.3, "C01": 0.8},
        md.yeoh: {"C10": 0.5, "C20": 0.1, "C30": 0.01},
        md.third_order_deformation: {
            "C10": 0.3,
            "C01": 0.2,
            "C11": 0.02,
            "C20": -0.1,
            "C30": 0.02,
        },
        md.ogden: {"mu": (1.0, 0.2), "alpha": (2.0, -1.5)},
        md.arruda_boyce: {"C1": 1.0, "limit": 3.2},
        md.extended_tube: {"Gc": 0.1867, "Ge": 0.2169, "beta": 0.2, "delta": 0.09693},
        md.van_der_waals: {"mu": 1.0, "beta": 0.1, "a": 0.5, "limit": 5.0},
        md.microsphere.affine.stretch: {
            "quadrature": q,
            "f": md.microsphere.langevin,
            "kwargs": {"mu": 1, "N": 10},
        },
        md.microsphere.affine.tube: {
            "quadrature": q,
            "f": md.microsphere.langevin2,
            "kwargs": {"mu": 1, "N": 10},
        },
        md.microsphere.nonaffine.stretch: {
            "quadrature": q,
            "p": 2.7,
            "f": md.microsphere.gauss,
            "kwargs": {"mu": 1},
        },
        md.microsphere.nonaffine.tube: {
            "quadrature": q,
            "q": 2.7,
            "f": md.microsphere.linear,
            "kwargs": {"mu": 1},
        },
        matadi.models.miehe_goektepe_lulei: {
            "mu": 0.1475,
            "N": 3.273,
            "p": 9.31,
            "U": 9.94,
            "q": 0.567,
        },
        matadi.models.linear_elastic: {
            "mu": 1.0,
            "lmbda": 2.0,
        },
    }

    return database


def test_models():

    # data
    np.random.seed(2345537)
    dF = np.random.rand(3, 3, 5, 100) - 0.5
    DF = np.random.rand(3, 3, 5, 100) - 0.5
    FF = np.random.rand(3, 3, 5, 100) - 0.5
    for a in range(3):
        FF[a, a] += 1

    pp = np.random.rand(5, 100)
    dp = np.random.rand(5, 100)
    Dp = np.random.rand(5, 100)

    JJ = 1 + np.random.rand(5, 100) / 10
    dJ = np.random.rand(5, 100) / 10
    DJ = np.random.rand(5, 100) / 10

    lib = library()

    for model, kwargs in lib.items():

        if model not in [md.saint_venant_kirchhoff, md.linear_elastic]:
            kwargs["bulk"] = 5000.0

        HM = matadi.MaterialHyperelastic(model, **kwargs)
        HM_mixed = matadi.ThreeFieldVariation(HM)
        HM_mixed2 = matadi.TwoFieldVariation(HM)

        W = HM.function([FF])
        P = HM.gradient([FF], threads=1)
        A = HM.hessian([FF])

        dW = HM.gradient_vector_product([FF], [dF])
        DW = HM.hessian_vector_product([FF], [dF], [DF])

        assert len(W) == 1
        assert len(P) == 1 + 1
        assert len(A) == 1

        dW_check = np.einsum("ij...,ij...->...", P[0], dF)
        DW_check = np.einsum("ij...,ijkl...,kl...->...", dF, A[0], DF)

        assert np.allclose(dW_check, dW[0])
        assert np.allclose(DW_check, DW[0])

        W = HM_mixed.function([FF, pp, JJ])
        P = HM_mixed.gradient([FF, pp, JJ])
        A = HM_mixed.hessian([FF, pp, JJ])

        assert len(W) == 1
        assert len(P) == 3 + 1
        assert len(A) == 6

        dW = HM_mixed.gradient_vector_product([FF, pp, JJ], [dF, dp, dJ])
        DW = HM_mixed.hessian_vector_product([FF, pp, JJ], [dF, dp, dJ], [DF, Dp, DJ])

        dWF = np.einsum("ij...,ij...->...", P[0], dF)
        dWp = P[1] * dp
        dWJ = P[2] * dJ

        assert np.allclose(dWF, dW[0])
        assert np.allclose(dWp, dW[1])
        assert np.allclose(dWJ, dW[2])

        dWdFdF = np.einsum("ij...,ijkl...,kl...->...", dF, A[0], DF)
        dWdpdp = dp * A[3][0, 0, 0, 0] * Dp
        dWdJdJ = dJ * A[5][0, 0, 0, 0] * DJ

        assert np.allclose(dWdFdF, DW[0])
        assert np.allclose(dWdpdp, DW[3])
        assert np.allclose(dWdJdJ, DW[5])

        dWdFdp = np.einsum("ij...,ij...,...->...", dF, A[1][:, :, 0, 0], Dp)
        dWdFdJ = np.einsum("ij...,ij...,...->...", dF, A[2][:, :, 0, 0], DJ)
        dWdpdJ = dp * A[4][0, 0, 0, 0] * DJ

        assert np.allclose(dWdFdF, DW[0])
        assert np.allclose(dWdFdp, DW[1])
        assert np.allclose(dWdFdJ, DW[2])
        assert np.allclose(dWdpdp, DW[3])
        assert np.allclose(dWdpdJ, DW[4])
        assert np.allclose(dWdJdJ, DW[5])

        W = HM_mixed2.function([FF, pp])
        P = HM_mixed2.gradient([FF, pp])
        A = HM_mixed2.hessian([FF, pp])

        assert len(W) == 1
        assert len(P) == 2 + 1
        assert len(A) == 3

        dW = HM_mixed2.gradient_vector_product([FF, pp], [dF, dp])
        DW = HM_mixed2.hessian_vector_product([FF, pp], [dF, dp], [DF, Dp])

        dWF = np.einsum("ij...,ij...->...", P[0], dF)
        dWp = P[1] * dp

        # exclude van-der-waals model from gvp- and hvp-checks
        # for two-field-variations due to nan-issues
        if model not in [md.van_der_waals]:
            assert np.allclose(dWF, dW[0])
        assert np.allclose(dWp, dW[1])

        dWdFdF = np.einsum("ij...,ijkl...,kl...->...", dF, A[0], DF)
        dWdpdp = dp * A[2][0, 0, 0, 0] * Dp

        if model not in [md.van_der_waals]:
            assert np.allclose(dWdFdF, DW[0])
        assert np.allclose(dWdpdp, DW[2])

        dWdFdp = np.einsum("ij...,ij...,...->...", dF, A[1][:, :, 0, 0], Dp)

        assert np.allclose(dWdFdp, DW[1])
        assert np.allclose(dWdpdp, DW[2])

    nh = matadi.MaterialHyperelastic(md.neo_hooke, **lib[md.neo_hooke])
    mr = matadi.MaterialHyperelastic(md.mooney_rivlin, **lib[md.mooney_rivlin])

    comp = matadi.MaterialComposite([nh, mr])
    comp.fun()

    W = comp.function([FF])
    P = comp.gradient([FF])
    A = comp.hessian([FF])

    assert len(W) == 1
    assert len(P) == 1 + 1
    assert len(A) == 1

    nh_mixed = matadi.ThreeFieldVariation(nh)
    mr_mixed = matadi.ThreeFieldVariation(mr)

    comp = matadi.MaterialComposite([nh_mixed, mr_mixed])

    W = comp.function([FF, pp, JJ])
    P = comp.gradient([FF, pp, JJ])
    A = comp.hessian([FF, pp, JJ])

    assert len(W) == 1
    assert len(P) == 3 + 1
    assert len(A) == 6

    nh_mixed2 = matadi.TwoFieldVariation(nh)
    mr_mixed2 = matadi.TwoFieldVariation(mr)

    comp = matadi.MaterialComposite([nh_mixed2, mr_mixed2])

    W = comp.function([FF, pp])
    P = comp.gradient([FF, pp])
    A = comp.hessian([FF, pp])

    assert len(W) == 1
    assert len(P) == 2 + 1
    assert len(A) == 3


if __name__ == "__main__":
    test_models()
