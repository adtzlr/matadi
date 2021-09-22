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
