from ..math import (
    transpose,
    det,
    DM,
    exp,
    cos,
    sin,
    pi,
    sqrt,
    if_else,
    invariants,
    trace,
)


def fiber(F, E, angle, k=1, axis=2, compression=False):
    "Fiber"

    a = angle * pi / 180
    plane = [(1, 2), (2, 0), (0, 1)][axis]

    N = DM.zeros(3)
    N[plane, :] = DM([cos(a), sin(a)])

    C = transpose(F) @ F
    Cu = det(C) ** (-1 / 3) * C

    stretch = sqrt((transpose(N) @ Cu @ N))[0, 0]
    strain = 1 / k * (stretch ** k - 1)

    if not compression:
        strain = if_else(strain < 0, 0, strain)

    return E * strain ** 2 / 2


def fiber_family(F, E, angle, k=1, axis=2, compression=False):
    "Fiber Family"

    f1 = fiber(F, E, k, angle, axis, compression)
    f2 = fiber(F, E, k, -angle, axis, compression)

    return f1 + f2


def holzapfel_gasser_ogden(F, c, k1, k2, kappa, angle, bulk=None, axis=2):
    "Holzapfel-Gasser-Ogden"

    if bulk is None:
        bulk = 5000.0 * c

    C = transpose(F) @ F
    J1, J2, J3 = invariants(C)
    I1 = J3 ** (-1 / 3) * J1

    alpha = angle * pi / 180
    plane = [(1, 2), (2, 0), (0, 1)][axis]

    N1 = DM.zeros(3)
    N1[plane, :] = DM([cos(alpha), sin(alpha)])

    N2 = DM.zeros(3)
    N2[plane, :] = DM([cos(alpha), -sin(alpha)])

    A1 = N1 @ transpose(N1)
    A2 = N2 @ transpose(N2)

    I4 = trace(J3 ** (-1 / 3) * C @ A1)
    I6 = trace(J3 ** (-1 / 3) * C @ A2)

    W_iso = c / 2 * (I1 - 3)

    w4 = exp(k2 * (kappa * I1 + (1 - 3 * kappa) * I4 - 1) ** 2) - 1
    w6 = exp(k2 * (kappa * I1 + (1 - 3 * kappa) * I6 - 1) ** 2) - 1

    W_aniso = k1 / (2 * k2) * (w4 + w6)

    return W_iso + W_aniso + bulk * (sqrt(J3) - 1) ** 2 / 2
