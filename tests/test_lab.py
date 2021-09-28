from matadi import MaterialHyperelastic, Lab
from matadi.models import neo_hooke, extended_tube, van_der_waals, mooney_rivlin

import matplotlib.pyplot as plt

from test_models import library


def pre(
    model,
    num=20,
    test_without_bulk=False,
    stability=False,
    plot=False,
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
        data = lab.run(ux=False, bx=False, ps=False, num=num, **run_kwargs)
        data = lab.run(ux=True, bx=False, ps=False, num=num, **run_kwargs)
        data = lab.run(ux=True, bx=True, ps=False, num=num, **run_kwargs)

    data = lab.run(ux=True, bx=True, ps=True, num=num, **run_kwargs)

    if plot:
        # plot stress vs. stretch
        fig, ax = lab.plot(data, stability=stability)

        if close:
            plt.close(fig)

    # dW and DW are lists
    assert len(data[0].stress) == num
    assert len(data[0].stretch) == num

    return data


def test_lab():

    data = pre(neo_hooke, test_without_bulk=True)
    data = pre(neo_hooke, run_all=True)

    run_kwargs = {"stretch_min": 0.1, "stretch_max": 1.0}

    data = pre(neo_hooke, run_kwargs=run_kwargs)
    data = pre(extended_tube)
    data = pre(van_der_waals)
    data = pre(mooney_rivlin, close=True, stability=True, plot=True)

    del data


if __name__ == "__main__":
    test_lab()
