import numpy as np

from matadi import Variable, MaterialTensor
from matadi.models import displacement_pressure_split

from matadi.math import det, dev, inv


def test_up():

    F = Variable("F", 3, 3)

    @displacement_pressure_split
    def fun(x):
        F = x[0]
        C = F.T @ F
        return det(F) ** (-2 / 3) * dev(C) @ inv(C)

    M = MaterialTensor([F, fun.p], fun, triu=True)

    FF = np.random.rand(3, 3, 8, 100)
    pp = np.random.rand(1, 8, 100)

    for a in range(3):
        FF[a, a] += 1

    P = M.function([FF, pp])
    A = M.gradient([FF, pp])

    assert len(P) == 2
    assert len(A) == 3


if __name__ == "__main__":
    test_up()
