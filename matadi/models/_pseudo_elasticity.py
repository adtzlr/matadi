from ..math import fmax, erf


def ogden_roxburgh(W, Wmaxn, r, m, beta):
    """An isotropic pseudo-elastic material formulation for the description of
    the Mullins-softening of rubber-like materials according to
    Ogden & Roxburgh."""

    # update the maximum (isochoric part of the) strain energy density
    # of the total load history (state variable)
    Wmax = fmax(W, Wmaxn)

    # evaluate the relative softening `eta`
    eta = 1 - 1 / r * erf((Wmax - W) / (m + beta * Wmax))

    return eta, Wmax
