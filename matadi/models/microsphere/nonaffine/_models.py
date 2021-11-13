from ..._helpers import isochoric_volumetric_split
from ....math import transpose, sum1, diag, sqrt, inv, det
from ..quadrature._bazant_oh import BazantOh


@isochoric_volumetric_split
def microsphere_nonaffine_stretch(F, p, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    C = transpose(F) @ F
    stretch = sum1(sqrt(diag(r.T @ C @ r)) ** p * w) ** (1 / p)

    return f(stretch, **kwargs)


@isochoric_volumetric_split
def microsphere_nonaffine_tube(F, q, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Non-affine tube part."

    r = quadrature.points
    w = quadrature.weights

    Fs = det(F) * transpose(inv(F))
    Cs = transpose(Fs) @ Fs
    areastretch = sum1(sqrt(diag(r.T @ Cs @ r)) ** q * w)

    return f(areastretch, **kwargs)
