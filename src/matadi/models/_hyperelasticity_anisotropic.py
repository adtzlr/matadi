from ..math import (
    DM,
    cos,
    exp,
    if_else,
    invariants,
    log,
    pi,
    sin,
    sqrt,
    trace,
    transpose,
    if_else
)
from ._helpers import isochoric_volumetric_split


@isochoric_volumetric_split
def fiber(F, E, angle, k=1, axis=2, compression=False):
    "Fiber"

    a = angle * pi / 180
    plane = [(1, 2), (2, 0), (0, 1)][axis]

    N = DM.zeros(3)
    N[plane, :] = DM([cos(a), sin(a)])

    C = transpose(F) @ F

    stretch = sqrt((transpose(N) @ C @ N))[0, 0]

    if k == 0:
        strain = log(stretch)
    else:
        strain = 1 / k * (stretch**k - 1)

    if not compression:
        strain = if_else(strain < 0, 0, strain)

    return E * strain**2 / 2


@isochoric_volumetric_split
def fiber_family(F, E, angle, k=1, axis=2, compression=False):
    "Fiber Family"

    f1 = fiber(F, E, angle=angle, k=k, axis=axis, compression=compression)
    f2 = fiber(F, E, angle=-angle, k=k, axis=axis, compression=compression)

    return f1 + f2


@isochoric_volumetric_split
def holzapfel_gasser_ogden(F, c, k1, k2, kappa, angle, axis=2):
    "Holzapfel-Gasser-Ogden"

    C = transpose(F) @ F
    I1, I2, I3 = invariants(C)

    alpha = angle * pi / 180
    plane = [(1, 2), (2, 0), (0, 1)][axis]

    N1 = DM.zeros(3)
    N1[plane, :] = DM([cos(alpha), sin(alpha)])

    N2 = DM.zeros(3)
    N2[plane, :] = DM([cos(alpha), -sin(alpha)])

    A1 = N1 @ transpose(N1)
    A2 = N2 @ transpose(N2)

    I4 = trace(C @ A1)
    I6 = trace(C @ A2)

    W_iso = c / 2 * (I1 - 3)

    E4 = kappa * I1 + (1 - 3 * kappa) * I4
    E6 = kappa * I1 + (1 - 3 * kappa) * I6

    # Tension-only fiber response
    w4 = if_else(E4 > 1, exp(k2 * (E4 - 1)**2) - 1, 0)
    w6 = if_else(E6 > 1, exp(k2 * (E6 - 1)**2) - 1, 0)

    W_aniso = k1 / (2 * k2) * (w4 + w6)

    return W_iso + W_aniso
