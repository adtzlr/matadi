from ...math import sqrt, atanh


def langevin(stretch, mu, N):
    """Langevin model (Padé approximation) given by the free energy
    of a single chain as a function of the stretch."""

    return mu * (stretch + 2 * sqrt(N) * atanh(stretch / sqrt(N)))


def gauss(stretch, mu):
    """Gaussian model given by the free energy
    of a single chain as a function of the stretch."""

    return 3 * mu / 2 * (stretch ** 2 - 1)
