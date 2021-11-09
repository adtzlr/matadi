from ..._helpers import isochoric_volumetric_split
from ....math import transpose, sum1, diag, sqrt, inv


@isochoric_volumetric_split
def microsphere_nonaffine_stretch(F, quadrature, p, f, kwargs):
    "Micro-sphere model: Non-affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    C = transpose(F) @ F
    stretch = sum1(sqrt(diag(r.T @ C @ r)) ** p * w) ** (1 / p)

    return f(stretch, **kwargs)


@isochoric_volumetric_split
def microsphere_nonaffine_tube(F, quadrature, q, f, kwargs):
    "Micro-sphere model: Non-affine tube part."

    r = quadrature.points
    w = quadrature.weights

    C = transpose(F) @ F
    areastretch = sum1(sqrt(diag(r.T @ inv(C) @ r)) ** q * w) ** (1 / q)

    return f(areastretch, **kwargs)
