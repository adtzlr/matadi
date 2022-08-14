import numpy as np

from matadi.models import NeoHookeOgdenRoxburgh, Morph


def test_up_templates():

    for MaterialUP in [NeoHookeOgdenRoxburgh, Morph]:

        # Material as a function of `F` and `p`
        # with additional state variables `z`
        M = MaterialUP()

        FF = (np.random.rand(3, 3, 8, 100) - 0.5) / 2
        pp = np.random.rand(1, 8, 100)
        zz = np.random.rand(*M.x[-1].shape, 8, 100)

        for a in range(3):
            FF[a, a] += 1

        P = M.function([FF, pp, zz])  # stress, constraint, statevars_new
        A = M.gradient([FF, pp, zz])

        assert len(P) == 3
        assert len(A) == 3


if __name__ == "__main__":
    test_up_templates()
