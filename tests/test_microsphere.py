import numpy as np

from matadi import MaterialTensor, Variable
from matadi.models.microsphere.affine import force


def nh(stretch, stretch_n, statevars_n, mu=1.0):
    "1d-Linear force-stretch consitututive material formulation."

    return 3 * mu * stretch, statevars_n


def test_microsphere_force():

    param = [nh, 1.0]

    F = Variable("F", 3, 3)
    Fn = Variable("Fn", 3, 3)
    Zn = Variable("Zn", 2, 21)

    umat = MaterialTensor(
        x=[F, force.p, Fn, Zn],
        fun=force,
        args=param,
        kwargs={"bulk": 5000},
        statevars=2,
        triu=True,
    )

    F = np.random.rand(3, 3, 8, 100) / 2
    Fn = np.random.rand(3, 3, 8, 100) / 2
    p = np.random.rand(1, 8, 100)
    Zn = np.random.rand(2, 21, 8, 100)

    for a in range(3):
        F[a, a] += 1
        Fn[a, a] += 1

    P, Q, Z = umat.function([F, p, Fn, Zn])
    A = umat.gradient([F, p, Fn, Zn])

    assert P.shape == (3, 3, 8, 100)
    assert Q.shape == (1, 1, 8, 100)
    assert Z.shape == (2, 21, 8, 100)

    assert len(A) == 3


if __name__ == "__main__":
    test_microsphere_force()
