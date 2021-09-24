from matadi import MaterialHyperelastic, Lab
from matadi.models import third_order_deformation

import matplotlib.pyplot as plt


def test_lab():

    mat = MaterialHyperelastic(
        third_order_deformation,
        C10=0.5,
        C01=0.1,
        C11=0.02,
        C20=-0.05,
        C30=0.01,
        bulk=5000.0,
    )

    lab = Lab(mat)
    data = lab.run(ux=False, bx=False, ps=False, num=20)
    data = lab.run(ux=True, bx=False, ps=False, num=20)
    data = lab.run(ux=True, bx=True, ps=False, num=20)
    fig, ax = lab.plot(data)

    plt.close(fig)

    # dW and DW are always lists...
    assert len(data[0].stress) == 50
    assert len(data[0].stretch) == 50


if __name__ == "__main__":
    test_lab()
