from ..._helpers import isochoric_volumetric_split
from ....math import transpose, sum1, diag, sqrt, inv


@isochoric_volumetric_split
def microsphere_affine_stretch(F, quadrature, f, kwargs):
    "Micro-sphere model: Non-affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    C = transpose(F) @ F
    stretch = sqrt(diag(r.T @ C @ r))

    return sum1(f(stretch, **kwargs) * w)


@isochoric_volumetric_split
def microsphere_affine_tube(F, quadrature, f, kwargs):
    "Micro-sphere model: Non-affine tube part."

    r = quadrature.points
    w = quadrature.weights

    C = transpose(F) @ F
    areastretch = sqrt(diag(r.T @ inv(C) @ r))

    return sum1(f(areastretch, **kwargs) * w)
