from multiprocessing import cpu_count

import numpy as np

from . import Material, Variable
from .math import det, horzcat, vertcat, zeros


class ThreeFieldVariation:
    def __init__(self, material):
        self.material = material
        p = Variable("p", 1, 1)
        J = Variable("J", 1, 1)
        self.x = [self.material.x[0], p, J]
        self.W = Material(self.x, self._fun)
        self.function = self.W.function
        self.gradient = self.W.gradient
        self.hessian = self.W.hessian

    def _fun(self, x):
        F, p, J = x
        detF = det(F)
        Fmod = (J / detF) ** (1 / 3) * F
        return self.material.fun(Fmod, **self.material.kwargs) + p * (detF - J)


class ThreeFieldVariationPlaneStrain:
    def __init__(self, material):
        self.material = material
        p = Variable("p", 1, 1)
        J = Variable("J", 1, 1)
        self.x = [self.material.x[0], p, J]
        self.W = Material(self.x, self._fun)
        self.function = self.W.function
        self.gradient = self.W.gradient
        self.hessian = self.W.hessian

    def _fun(self, x):
        F, p, J = x
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        F[2, 2] = 1  # fixed thickness ratio `h / H = 1`
        detF = det(F)
        Fmod = (J / detF) ** (1 / 3) * F
        return self.material.fun(Fmod, **self.material.kwargs) + p * (detF - J)


class MaterialHyperelastic:
    def __init__(self, fun, **kwargs):
        F = Variable("F", 3, 3)
        self.x = [F]
        self.fun = fun
        self.kwargs = kwargs
        self.W = Material(self.x, self._fun_wrapper, kwargs=self.kwargs)
        self.function = self.W.function
        self.gradient = self.W.gradient
        self.hessian = self.W.hessian

    def _fun_wrapper(self, x, **kwargs):
        return self.fun(x[0], **kwargs)


class MaterialHyperelasticPlaneStrain:
    def __init__(self, fun, **kwargs):
        F = Variable("F", 2, 2)
        self.x = [F]
        self.fun = fun
        self.kwargs = kwargs
        self.W = Material(self.x, self._fun_wrapper, kwargs=self.kwargs)
        self.function = self.W.function
        self.gradient = self.W.gradient
        self.hessian = self.W.hessian

    def _fun_wrapper(self, x, **kwargs):
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        F[2, 2] = 1  # fixed thickness ratio `h / H = 1`
        return self.fun(F, **kwargs)


class MaterialHyperelasticPlaneStressIncompressible(MaterialHyperelasticPlaneStrain):
    def __init__(self, fun, **kwargs):
        super().__init__(fun, **kwargs)

    def _fun_wrapper(self, x, **kwargs):
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        F[2, 2] = 1 / det(x[0])  # thickness ratio `h / H = 1 / (a / A)`
        return self.fun(F, **kwargs)


class MaterialHyperelasticPlaneStressLinearElastic(MaterialHyperelasticPlaneStrain):
    def __init__(self, fun, **kwargs):
        super().__init__(fun, **kwargs)

    def _fun_wrapper(self, x, **kwargs):
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        # stress-free thickness ratio for linear elastic material
        # s_33 != 0 = 2 mu e_33 + lmbda (e_11 + e_22 + e_33)
        # e_33 = - (e_11 + e_22) * lmbda / (2 mu + lmbda)
        # F_33 = 1 + e_33
        F[2, 2] = 1 - (F[0, 0] + F[1, 1] - 2) * (
            kwargs["lmbda"] / (2 * kwargs["mu"] + kwargs["lmbda"])
        )
        return self.fun(F, **kwargs)


class MaterialComposite:
    def __init__(self, materials):
        "Composite Material as a sum of a list of materials."
        self.materials = materials
        self.fun = self.composite

    def composite(self):
        "Dummy function for plot title."
        return

    def function(self, x, **kwargs):
        fun = [m.function(x, **kwargs) for m in self.materials]
        return [np.sum([f[a] for f in fun], 0) for a in range(len(fun[0]))]

    def gradient(self, x, **kwargs):
        grad = [m.gradient(x, **kwargs) for m in self.materials]
        return [np.sum([g[a] for g in grad], 0) for a in range(len(grad[0]))]

    def hessian(self, x, **kwargs):
        hess = [m.hessian(x, **kwargs) for m in self.materials]
        return [np.sum([h[a] for h in hess], 0) for a in range(len(hess[0]))]
