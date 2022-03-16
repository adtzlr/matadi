from functools import wraps
from copy import deepcopy

from .. import Variable
from ..math import det, cof, trace, gradient


def isochoric_volumetric_split(fun):
    """Apply the material formulation only on the isochoric part of the
    multiplicative split of the deformation gradient. Optionally, if
    the bulk keyword is passed, add a volumetric term."""

    @wraps(fun)
    def apply_iso(*args, **kwargs):
        F = args[0]
        J = det(F)
        F_iso = J ** (-1 / 3) * F

        fun_args = args[1:]
        fun_kwargs = deepcopy(kwargs)

        if "bulk" in kwargs.keys():
            _ = fun_kwargs.pop("bulk")

        W = fun(F_iso, *fun_args, **fun_kwargs)

        if "bulk" in kwargs.keys():
            W += volumetric(J, kwargs["bulk"])

        return W

    return apply_iso


def volumetric(J, bulk):
    return bulk * (J - 1) ** 2 / 2


def displacement_pressure_split(fun):
    """Apply the (u/p)-framework on top of a material formulation (a function
    of the deformation gradient which returns the first Piola-Kirchhoff stress)
    . The additional hydrostatic stress variable `p` is attached as an
    attribute `fun.p` to the augmented function."""

    p = Variable("p")

    @wraps(fun)
    def apply_up(*args, **kwargs):

        F = args[0][0]

        f = fun(*args, **kwargs)

        # check if function is list or tuple
        if not (isinstance(f, list) or isinstance(f, tuple)):
            f = [f]

        P = f[0]
        P_vol = trace(P @ F.T) / det(F) / 3
        K = trace(gradient(P_vol, F) @ F.T) / det(F) / 3

        return [P - (P_vol - p) * cof(F), (P_vol - p) / K, *f[1:]]

    apply_up.p = p

    return apply_up
