import numpy as np

import matadi
import matadi.models as md


def get_title(model):
    return "-".join([m.title() for m in model.__name__.split("_")])


def library():
    "Library with models and parameters."

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
    }

    return database


def test_models():

    # data
    np.random.seed(2345537)
    FF = np.random.rand(3, 3, 5, 100)
    for a in range(3):
        FF[a, a] += 1

    pp = np.random.rand(5, 100)
    JJ = 1 + np.random.rand(5, 100) / 10

    lib = library()

    for model, kwargs in lib.items():

        if model is not md.saint_venant_kirchhoff:
            kwargs["bulk"] = 5000.0

        HM = matadi.MaterialHyperelastic(model, **kwargs)
        HM_mixed = matadi.ThreeFieldVariation(HM)

        W = HM.function([FF])
        P = HM.gradient([FF])
        A = HM.hessian([FF])

        assert len(W) == 1
        assert len(P) == 1
        assert len(A) == 1

        W = HM_mixed.function([FF, pp, JJ])
        P = HM_mixed.gradient([FF, pp, JJ])
        A = HM_mixed.hessian([FF, pp, JJ])

        assert len(W) == 1
        assert len(P) == 3
        assert len(A) == 6


if __name__ == "__main__":
    test_models()
