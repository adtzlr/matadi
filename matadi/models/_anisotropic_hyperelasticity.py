from ..math import transpose, DM, cos, sin, pi, sqrt, if_else


def fiber(F, E, k=1, angle=30, axis=2, compression=False):

    a = angle * pi / 180
    plane = [(1, 2), (2, 0), (0, 1)][axis]

    N = DM.ones(3)
    N[plane, :] = DM([cos(a), sin(a)])

    C = transpose(F) @ F

    stretch = sqrt((transpose(N) @ C @ N))[0]
    strain = 1 / k * (stretch ** k - 1)

    if not compression:
        strain = if_else(strain < 0, 0, strain)

    return E * strain


def fiber_family(F, E, k=1, angle=30, axis=2, compression=False):

    f1 = fiber(F, E, k, angle, axis, compression)
    f2 = fiber(F, E, k, -angle, axis, compression)

    return f1 + f2
