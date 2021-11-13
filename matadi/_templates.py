import numpy as np

from . import Material, Variable
from .math import det


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


class MaterialComposite:
    def __init__(self, materials):
        "Composite Material as a sum of a list of materials."
        self.materials = materials
        self.fun = self.composite

    def composite(self):
        "Dummy function for plot title."
        return

    def function(self, x):
        fun = [m.function(x) for m in self.materials]
        return [np.sum([f[a] for f in fun], 0) for a in range(len(fun[0]))]

    def gradient(self, x):
        grad = [m.gradient(x) for m in self.materials]
        return [np.sum([g[a] for g in grad], 0) for a in range(len(grad[0]))]

    def hessian(self, x):
        hess = [m.hessian(x) for m in self.materials]
        return [np.sum([h[a] for h in hess], 0) for a in range(len(hess[0]))]
