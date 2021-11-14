from ..._helpers import isochoric_volumetric_split
from ....math import transpose, sum1, diag, sqrt, inv, det
from ..quadrature._bazant_oh import BazantOh


@isochoric_volumetric_split
def microsphere_affine_stretch(F, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine stretch part."

    r = quadrature.points
    w = quadrature.weights

    C = transpose(F) @ F
    affine_stretch = sqrt(diag(r.T @ C @ r))

    return sum1(f(affine_stretch, **kwargs) * w)


@isochoric_volumetric_split
def microsphere_affine_tube(F, f, kwargs, quadrature=BazantOh(n=21)):
    "Micro-sphere model: Affine area-stretch part."

    r = quadrature.points
    w = quadrature.weights

    Fs = det(F) * transpose(inv(F))
    Cs = transpose(Fs) @ Fs
    affine_areastretch = sqrt(diag(r.T @ Cs @ r))

    return sum1(f(affine_areastretch, **kwargs) * w)
