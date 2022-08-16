from ..._helpers import isochoric_volumetric_split, displacement_pressure_split
from ....math import transpose, sum1, diag, sqrt, inv, det, reshape, dev
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


@displacement_pressure_split
def microsphere_affine_force(x, f, *args, **kwargs):
    """Micro-sphere model: Forces of affine stretch model as first Piola-
    Kirchhoff stress tensor embedded into a (u/p)-framework."""

    # sphere quadrature
    sphere = BazantOh(n=21)

    # extract current and initial deformation gradient and state variables
    F = x[0]
    statevars_n = x[-1]

    # volume ratios
    J = det(F)

    # unimodular part of current and initial right Cauchy-Green deformation tensor
    C = F.T @ F
    CG = J ** (-2 / 3) * (C)

    # affine stretches
    lam = sqrt(diag(sphere.points.T @ CG @ sphere.points))

    bulk = kwargs.pop("bulk")

    # fiber forces and state variable update
    f, statevars = f(lam, statevars_n, *args, **kwargs)

    # Second Piola-Kirchhoff stress tensor
    SG = reshape(sum1(f / lam * sphere.weights * sphere.bases), 3, 3)
    S = dev(SG @ CG) @ inv(C) + bulk * (J - 1) * J * inv(C)

    return F @ S, statevars
