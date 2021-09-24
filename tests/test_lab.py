from matadi import MaterialHyperelastic, Lab
from matadi.models import neo_hooke

import matplotlib.pyplot as plt


def test_lab():

    mat = MaterialHyperelastic(
        neo_hooke,
        C10=0.5,
        bulk=20.0,
    )

    lab = Lab(mat)
    data = lab.run(ux=False, bx=False, ps=False, num=20)
    data = lab.run(ux=True, bx=False, ps=False, num=20)
    data = lab.run(ux=True, bx=True, ps=False, num=20)
    data = lab.run(ux=True, bx=True, ps=True, num=20)
    fig, ax = lab.plot(data)

    #plt.close(fig)

    # dW and DW are always lists...
    assert len(data[0].stress) == 20
    assert len(data[0].stretch) == 20


if __name__ == "__main__":
    test_lab()
