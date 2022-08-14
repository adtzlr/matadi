import numpy as np

from matadi import Variable, MaterialTensor
from matadi.models import (
    morph
)


def test_up_morph():

    # deformation gradient
    F = Variable("F", 3, 3)

    # state variables
    z = Variable("z", 13, 1)

    kwargs = {
        "param": [0.035, 0.37, 0.17, 2.4, 0.01, 6.4, 5.5, 0.24],
        "bulk": 5000
    }

    p = morph.p
    M = MaterialTensor(x=[F, p, z], fun=morph, triu=True, statevars=1, kwargs=kwargs)

    FF = np.random.rand(3, 3, 8, 100)
    pp = np.random.rand(1, 8, 100)
    zz = np.random.rand(13, 1, 8, 100)

    for a in range(3):
        FF[a, a] += 1

    P = M.function([FF, pp, zz])  # stress, constraint, statevars_new
    A = M.gradient([FF, pp, zz])

    assert len(P) == 3
    assert len(A) == 3


if __name__ == "__main__":
    test_up_morph()