import numpy as np

from . import Material, MaterialTensor, Variable
from .math import det, horzcat, vertcat, zeros, gradient as grad, trace, eye, Function


class TwoFieldVariation:
    def __init__(self, material):
        self.material = material
        p = Variable("p", 1, 1)
        self.x = [self.material.x[0], p]
        self.W = Material(self.x, self._fun)
        self.function = self.W.function
        self.hessian = self.W.hessian
        self.gradient_vector_product = self.W.gradient_vector_product
        self.hessian_vector_product = self.W.hessian_vector_product

    def _fun(self, x):
        F, p = x[:2]
        J = det(F)
        W = self.material.fun(F, **self.material.kwargs)
        U = self.material.fun(J ** (1 / 3) * eye(3), **self.material.kwargs)
        dWdF = grad(W, F)
        dWdJ = trace(dWdF @ F.T) / J / 3
        d2WdJdJ = trace(grad(dWdJ, F) @ F.T) / J / 3
        return W - U + 1 / d2WdJdJ * (p * dWdJ - p**2 / 2)

    def gradient(self, x, *args, **kwargs):
        return [*self.W.gradient(x, *args, **kwargs), None]


class TwoFieldVariationPlaneStrain:
    def __init__(self, material):
        self.material = material
        p = Variable("p", 1, 1)
        self.x = [self.material.x[0], p]
        self.W = Material(self.x, self._fun)
        self.function = self.W.function
        self.hessian = self.W.hessian
        self.gradient_vector_product = self.W.gradient_vector_product
        self.hessian_vector_product = self.W.hessian_vector_product

    def _fun(self, x):
        F, p = x[:2]
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        F[2, 2] = 1  # fixed thickness ratio `h / H = 1`
        J = det(F)
        W = self.material.fun(F, **self.material.kwargs)
        U = self.material.fun(J ** (1 / 3) * eye(3), **self.material.kwargs)

        f = Variable("f", 3, 3)
        w = self.material.fun(f, **self.material.kwargs)

        dwdf = grad(w, f)
        dwdj = trace(dwdf @ f.T) / det(f) / 3
        d2wdjdj = trace(grad(dwdj, f) @ f.T) / det(f) / 3

        dWdJ = Function("w", [f], [dwdj])(F)
        d2WdJdJ = Function("w", [f], [d2wdjdj])(F)

        return W - U + 1 / d2WdJdJ * (p * dWdJ - p**2 / 2)

    def gradient(self, x, *args, **kwargs):
        return [*self.W.gradient(x, *args, **kwargs), None]


class ThreeFieldVariation:
    def __init__(self, material):
        self.material = material
        p = Variable("p", 1, 1)
        J = Variable("J", 1, 1)
        self.x = [self.material.x[0], p, J]
        self.W = Material(self.x, self._fun)
        self.function = self.W.function
        self.hessian = self.W.hessian
        self.gradient_vector_product = self.W.gradient_vector_product
        self.hessian_vector_product = self.W.hessian_vector_product

    def _fun(self, x):
        F, p, J = x[:3]
        detF = det(F)
        Fmod = (J / detF) ** (1 / 3) * F
        return self.material.fun(Fmod, **self.material.kwargs) + p * (detF - J)

    def gradient(self, x, *args, **kwargs):
        return [*self.W.gradient(x, *args, **kwargs), None]


class ThreeFieldVariationPlaneStrain:
    def __init__(self, material):
        self.material = material
        p = Variable("p", 1, 1)
        J = Variable("J", 1, 1)
        self.x = [self.material.x[0], p, J]
        self.W = Material(self.x, self._fun)
        self.function = self.W.function
        self.hessian = self.W.hessian
        self.gradient_vector_product = self.W.gradient_vector_product
        self.hessian_vector_product = self.W.hessian_vector_product

    def _fun(self, x):
        F, p, J = x[:3]
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        F[2, 2] = 1  # fixed thickness ratio `h / H = 1`
        detF = det(F)
        Fmod = (J / detF) ** (1 / 3) * F
        return self.material.fun(Fmod, **self.material.kwargs) + p * (detF - J)

    def gradient(self, x, *args, **kwargs):
        return [*self.W.gradient(x, *args, **kwargs), None]


class MaterialHyperelastic:
    def __init__(self, fun, **kwargs):
        F = Variable("F", 3, 3)
        self.x = [F]
        self.fun = fun
        self.kwargs = kwargs
        self.W = Material(self.x, self._fun_wrapper, kwargs=self.kwargs)
        self.gradient_vector_product = self.W.gradient_vector_product
        self.hessian_vector_product = self.W.hessian_vector_product

    def _fun_wrapper(self, x, **kwargs):
        return self.fun(x[0], **kwargs)

    def function(self, x, *args, **kwargs):
        return self.W.function(x[:1], *args, **kwargs)

    def gradient(self, x, *args, **kwargs):
        return [*self.W.gradient(x[:1], *args, **kwargs), None]

    def hessian(self, x, *args, **kwargs):
        return self.W.hessian(x[:1], *args, **kwargs)


class MaterialHyperelasticPlaneStrain:
    def __init__(self, fun, **kwargs):
        F = Variable("F", 2, 2)
        self.x = [F]
        self.fun = fun
        self.kwargs = kwargs
        self.W = Material(self.x, self._fun_wrapper, kwargs=self.kwargs)
        self.gradient_vector_product = self.W.gradient_vector_product
        self.hessian_vector_product = self.W.hessian_vector_product

    def _fun_wrapper(self, x, **kwargs):
        F = horzcat(vertcat(x[0], zeros(1, 2)), zeros(3, 1))
        F[2, 2] = 1  # fixed thickness ratio `h / H = 1`
        return self.fun(F, **kwargs)

    def function(self, x, *args, **kwargs):
        return self.W.function(x[:1], *args, **kwargs)

    def gradient(self, x, *args, **kwargs):
        return [*self.W.gradient(x[:1], *args, **kwargs), None]

    def hessian(self, x, *args, **kwargs):
        return self.W.hessian(x[:1], *args, **kwargs)


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
        "Composite Material as a sum of a list of hyperelastic materials."
        self.materials = materials
        self.fun = self.composite

        # get number of variables defined in the first material
        self._n = len(self.materials[0].x)

    def composite(self):
        "Dummy function for plot title."
        return

    def function(self, x, **kwargs):
        fun = [m.function(x[: self._n], **kwargs) for m in self.materials]
        return [np.sum([f[a] for f in fun], 0) for a in range(len(fun[0]))]

    def gradient(self, x, **kwargs):
        grad = [m.gradient(x[: self._n], **kwargs)[: len(x)] for m in self.materials]
        res = [np.sum([g[a] for g in grad], 0) for a in range(len(grad[0]))]
        return [*res, None]

    def hessian(self, x, **kwargs):
        hess = [m.hessian(x[: self._n], **kwargs) for m in self.materials]
        return [np.sum([h[a] for h in hess], 0) for a in range(len(hess[0]))]


class MaterialTensorGeneral(MaterialTensor):
    def __init__(self, fun, statevars_shape=(1, 1), x=None, triu=True, **kwargs):
        """A (first Piola-Kirchhoff stress) tensor-based material definition with
        state variables of a given shape."""

        if x is None:
            x = [Variable("F", 3, 3)]

        try:
            # displacement-pressure split
            x.append(fun.p)
        except:
            pass

        # add state variables
        x.append(Variable("z", *statevars_shape))

        super().__init__(x=x, fun=fun, triu=triu, statevars=1, kwargs=kwargs)
