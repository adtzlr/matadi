import numpy as np

from matadi import Variable, MaterialTensor
from matadi.models import displacement_pressure_split

from matadi.math import det, dev, inv


def test_up():

    # deformation gradient
    F = Variable("F", 3, 3)
    
    # state variables
    z = Variable("z", 5, 9)

    @displacement_pressure_split
    def fun(x):
        F, z = x[0], x[-1]
        C = F.T @ F
        return det(F) ** (-2 / 3) * dev(C) @ inv(C), z
    
    # get pressure variable from augmented function
    p = fun.p

    # Materail as a function of `F` and `p`
    # with additional state variables `z`
    M = MaterialTensor([F, p, z], fun, triu=True, statevars=1)

    FF = np.random.rand(3, 3, 8, 100)
    pp = np.random.rand(1, 8, 100)
    zz = np.random.rand(5, 9, 8, 100)

    for a in range(3):
        FF[a, a] += 1

    P, constraint, statevars_new = M.function([FF, pp, zz])
    A = M.gradient([FF, pp, zz])

    assert len(P) == 3
    assert len(A) == 3


if __name__ == "__main__":
    test_up()
