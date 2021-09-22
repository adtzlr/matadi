from ..math import det, transpose, trace, eigvals, sum1


def neo_hooke(F, C10, bulk):
    J = det(F)
    C = transpose(F) @ F
    I1 = J ** (-2 / 3) * trace(C)
    return C10 * (I1 - 3) + bulk * (J - 1) ** 2 / 2


def mooney_rivlin(F, C10, C01, bulk):
    J = det(F)
    C = transpose(F) @ F
    I1 = J ** (-2 / 3) * trace(C)
    I2 = J ** (-4 / 3) * trace(C @ C)
    return C10 * (I1 - 3) + C01 * (I2 - 3) + bulk * (J - 1) ** 2 / 2


def yeoh(F, C10, C20, C30, bulk):
    J = det(F)
    C = transpose(F) @ F
    I1 = J ** (-2 / 3) * trace(C)
    return (
        C10 * (I1 - 3)
        + C20 * (I1 - 3) ** 2
        + C30 * (I1 - 3) ** 3
        + bulk * (J - 1) ** 2 / 2
    )


def third_order_deformation(F, C10, C01, C11, C20, C30, bulk):
    J = det(F)
    C = transpose(F) @ F
    I1 = J ** (-2 / 3) * trace(C)
    I2 = J ** (-4 / 3) * trace(C @ C)
    return (
        C10 * (I1 - 3)
        + C01 * (I2 - 3)
        + C11 * (I1 - 3) * (I2 - 3)
        + C20 * (I1 - 3) ** 2
        + C30 * (I1 - 3) ** 3
        + bulk * (J - 1) ** 2 / 2
    )


def ogden(F, mu, alpha, bulk):
    J = det(F)
    C = transpose(F) @ F
    wC = eigvals(J ** (-2 / 3) * C)

    out = 0
    for m, a in zip(mu, alpha):
        wk = wC ** (a / 2)
        out += m / a * (sum1(wk)[0, 0] - 3)

    return out + bulk * (J - 1) ** 2 / 2


def arruda_boyce(F, C1, limit, bulk):
    J = det(F)
    C = transpose(F) @ F
    I1 = trace(J ** (-2 / 3) * C)

    alpha = [1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750]
    beta = 1 / limit ** 2

    out = 0
    for i, a in enumerate(alpha):
        j = i + 1
        out += a * beta ** (2 * j - 2) * (I1 ** j - 3 ** j)

    return C1 * out + bulk * (J - 1) ** 2 / 2
