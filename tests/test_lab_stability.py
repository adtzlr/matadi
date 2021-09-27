from matadi import MaterialHyperelastic, Lab
from matadi.models import mooney_rivlin

import matplotlib.pyplot as plt


def test_lab_stab():

    mat = MaterialHyperelastic(mooney_rivlin, C10=0.0, C01=0.5, bulk=2000.0,)

    lab = Lab(mat)
    data = lab.run(ux=True, bx=True, ps=True, num=50, stretch_max=2.5)

    fig, ax = lab.plot(data, stability=True)
    plt.close(fig)

    # dW and DW are always lists...
    assert len(data[0].stress) == 50
    assert len(data[0].stretch) == 50

    data = lab.run(ux=True, bx=True, ps=True, num=50, stretch_min=0.1, stretch_max=1.0)

    fig, ax = lab.plot(data, stability=True)
    # plt.close(fig)

    # dW and DW are always lists...
    assert len(data[0].stress) == 50
    assert len(data[0].stretch) == 50


if __name__ == "__main__":
    test_lab_stab()
