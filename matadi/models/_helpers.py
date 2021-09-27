from functools import wraps
from copy import deepcopy

from ..math import det


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
