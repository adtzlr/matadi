from matadi import MaterialHyperelastic, Lab
import matadi.models as md
from matadi.models import neo_hooke, extended_tube, van_der_waals, mooney_rivlin

import matplotlib.pyplot as plt

import pytest


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


def pre(
    model,
    num=20,
    test_without_bulk=False,
    stability=False,
    plot=False,
    plot_shear=False,
    close=True,
    run_all=False,
    run_kwargs={},
):

    lib = library()
    kwargs = lib[model]

    if test_without_bulk:
        # init material without bulk modulus
        mat = MaterialHyperelastic(model, **kwargs)

    # init material with bulk modulus
    mat = MaterialHyperelastic(model, **kwargs, bulk=5000.0)

    # init lab
    lab = Lab(mat)

    # run experiments
    if run_all:
        data = lab.run(ux=False, bx=False, ps=False, shear=False, num=num, **run_kwargs)
        data = lab.run(ux=True, bx=False, ps=False, shear=False, num=num, **run_kwargs)
        data = lab.run(ux=True, bx=True, ps=False, shear=False, num=num, **run_kwargs)

        data = lab.run(ux=False, bx=False, ps=False, shear=True, num=num, **run_kwargs)
        data = lab.run(ux=True, bx=False, ps=False, shear=True, num=num, **run_kwargs)
        data = lab.run(ux=True, bx=True, ps=False, shear=True, num=num, **run_kwargs)

    data = lab.run(ux=True, bx=True, ps=True, shear=True, num=num, **run_kwargs)

    if plot:
        # plot stress vs. stretch
        fig, ax = lab.plot(data, stability=stability)

        if close:
            plt.close(fig)

    if plot_shear:
        # plot stress vs. stretch
        fig, ax = lab.plot_shear(data)

        if close:
            plt.close(fig)

    # dW and DW are lists
    assert len(data[0].stress) == num
    assert len(data[0].stretch) == num

    with pytest.raises(TypeError):
        data_x = lab.run(ux=True, bx=True, ps=True, shear=False, num=num, **run_kwargs)
        lab.plot_shear(data_x)

    return data


def test_lab():

    data = pre(neo_hooke, test_without_bulk=True)
    data = pre(neo_hooke, run_all=True, plot=True, plot_shear=True, close=True)

    run_kwargs = {"stretch_min": 0.1, "stretch_max": 1.0, "shear_max": 0.8}

    data = pre(neo_hooke, run_kwargs=run_kwargs)
    data = pre(extended_tube)
    data = pre(van_der_waals)
    data = pre(mooney_rivlin, close=True, stability=True, plot=True, plot_shear=True)

    del data


if __name__ == "__main__":
    test_lab()
