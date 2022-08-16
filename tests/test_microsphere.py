import numpy as np

from matadi import MaterialTensor, Variable
from matadi.models.microsphere import affine


def nh(stretch, statevars_n, mu=1.0):
    "1d-Linear force-stretch consitututive material formulation."

    return 3 * mu * stretch, statevars_n


def test_microsphere_force():

    F = Variable("F", 3, 3)
    Fn = Variable("Fn", 3, 3)
    Zn = Variable("Zn", 5, 21)

    umat = MaterialTensor(
        x=[F, affine.force.p, Zn],
        fun=affine.force,
        kwargs={"f": nh, "mu": 1.0, "bulk": 5000},
        statevars=1,
        triu=True,
    )

    F = np.random.rand(3, 3, 8, 100) / 2
    p = np.random.rand(1, 8, 100)
    Zn = np.random.rand(5, 21, 8, 100)

    for a in range(3):
        F[a, a] += 1
        Fn[a, a] += 1

    P, Q, Z = umat.function([F, p, Zn])
    A = umat.gradient([F, p, Zn])

    assert P.shape == (3, 3, 8, 100)
    assert Q.shape == (1, 1, 8, 100)
    assert Z.shape == (5, 21, 8, 100)

    assert len(A) == 3


if __name__ == "__main__":
    test_microsphere_force()
