from ..math import transpose, det, DM, exp, cos, sin, pi, sqrt, if_else


def fiber(F, E, angle, k=1, axis=2, compression=False):

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

    f1 = fiber(F, E, k, angle, axis, compression)
    f2 = fiber(F, E, k, -angle, axis, compression)

    return f1 + f2


def holzapfel_gasser_ogden(F, c, k1, k2, angle, axis=2):

    C = transpose(F) @ F
    J1, J2, J3 = invariants(C)

    Cu = J3 ** (-1 / 3) * C

    I1 = J3 ** (-1 / 3) * J1
    # I2 = J3 ** (-2 / 3) * J2

    alpha = angle * pi / 180
    plane = [(1, 2), (2, 0), (0, 1)][axis]

    N1 = DM.zeros(3)
    N1[plane, :] = DM([cos(alpha), sin(alpha)])

    N2 = DM.zeros(3)
    N2[plane, :] = DM([cos(alpha), -sin(alpha)])

    A1 = N1 @ transpose(N1)
    A2 = N2 @ transpose(N2)

    I4 = trace(Cu @ A1)
    I6 = trace(Cu @ A2)

    # I5 = trace(C @ C @ A1)
    # I7 = trace(C @ C @ A2)

    # I8 = (transpose(N1) @ N2) * transpose(N1) @ C @ N2

    W_iso = c / 2 * (I1 - 3)

    w1 = exp(k2 * (I4 - 1) ** 2) - 1
    w2 = exp(k2 * (I6 - 1) ** 2) - 1

    W_aniso = k1 / (2 * k2) * (w1 + w2)

    return W_iso + W_aniso
