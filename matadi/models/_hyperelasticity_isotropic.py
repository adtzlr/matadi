from ._helpers import isochoric_volumetric_split
from ..math import dot, det, transpose, trace, eigvals, sum1, log, sqrt, eye, sym


def linear_elastic(F, mu, lmbda):
    strain = sym(F - eye(3))
    return mu * trace(strain @ strain) + lmbda / 2 * trace(strain) ** 2


def saint_venant_kirchhoff(F, mu, lmbda):
    C = dot(transpose(F), F)
    I1 = trace(C) / 2 - 3 / 2
    I2 = trace(C @ C) / 4 - trace(C) / 2 + 3 / 4
    return mu * I2 + lmbda * I1 ** 2 / 2


@isochoric_volumetric_split
def neo_hooke(F, C10):
    C = transpose(F) @ F
    I1 = trace(C)
    return C10 * (I1 - 3)


@isochoric_volumetric_split
def mooney_rivlin(F, C10, C01):
    C = transpose(F) @ F
    I1 = trace(C)
    I2 = (trace(C) ** 2 - trace(C @ C)) / 2
    return C10 * (I1 - 3) + C01 * (I2 - 3)


@isochoric_volumetric_split
def yeoh(F, C10, C20, C30):
    J = det(F)
    C = transpose(F) @ F
    I1 = J ** (-2 / 3) * trace(C)
    return C10 * (I1 - 3) + C20 * (I1 - 3) ** 2 + C30 * (I1 - 3) ** 3


@isochoric_volumetric_split
def third_order_deformation(F, C10, C01, C11, C20, C30):
    C = transpose(F) @ F
    I1 = trace(C)
    I2 = (trace(C) ** 2 - trace(C @ C)) / 2
    return (
        C10 * (I1 - 3)
        + C01 * (I2 - 3)
        + C11 * (I1 - 3) * (I2 - 3)
        + C20 * (I1 - 3) ** 2
        + C30 * (I1 - 3) ** 3
    )


@isochoric_volumetric_split
def ogden(F, mu, alpha):
    C = transpose(F) @ F
    wC = eigvals(C)

    out = 0
    for m, a in zip(mu, alpha):
        wk = wC ** (a / 2)
        out += m / a * (sum1(wk)[0, 0] - 3)

    return out


@isochoric_volumetric_split
def arruda_boyce(F, C1, limit):
    C = transpose(F) @ F
    I1 = trace(C)

    alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    beta = 1 / limit ** 2

    out = 0
    for i, a in enumerate(alpha):
        j = i + 1
        out += a * beta ** (2 * j - 2) * (I1 ** j - 3 ** j)

    return C1 * out


@isochoric_volumetric_split
def extended_tube(F, Gc, delta, Ge, beta):
    C = transpose(F) @ F
    D = trace(C)
    wC = eigvals(C)
    g = (1 - delta ** 2) * (D - 3) / (1 - delta ** 2 * (D - 3))
    Wc = Gc / 2 * (g + log(1 - delta ** 2 * (D - 3)))
    We = 2 * Ge / beta ** 2 * sum1(wC ** (-beta / 2) - 1)
    return Wc + We


@isochoric_volumetric_split
def van_der_waals(F, mu, limit, a, beta):
    C = transpose(F) @ F
    I1 = trace(C)
    I2 = (trace(C) ** 2 - trace(C @ C)) / 2
    I = (1 - beta) * I1 + beta * I2
    eta = sqrt((I - 3) / (limit ** 2 - 3))
    return mu * (
        -(limit ** 2 - 3) * (log(1 - eta) + eta) - 2 / 3 * a * ((I - 3) / 2) ** (3 / 2)
    )
