from multiprocessing import cpu_count

import numpy as np
import casadi as ca

from ._apply import apply
from . import Variable


class Function:
    def __init__(self, x, fun, args=(), kwargs={}, compress=False):

        self.x = x
        self._fun = fun

        self.args = args
        self.kwargs = kwargs

        # generate function
        f = self._fun(self.x, *self.args, **self.kwargs)

        # check if function is list or tuple
        if isinstance(f, list) or isinstance(f, tuple):
            self._f = f
        else:
            self._f = [f]

        # generate casADi function objects
        self._function = ca.Function("f", self.x, self._f)

        # generate indices
        self._idx_function = [()]
        self._idx_gradient = [y.shape for y in x]

    def function(self, x, threads=cpu_count()):
        "Return the function."
        return apply(
            x,
            fun=self._function,
            x_shape=self._idx_gradient,
            fun_shape=self._idx_function,
            threads=threads,
        )


class FunctionTensor:
    def __init__(self, x, fun, args=(), kwargs={}, compress=False):

        self.x = x
        self._fun = fun

        self.args = args
        self.kwargs = kwargs

        # generate function
        f = self._fun(self.x, *self.args, **self.kwargs)

        # check if function is list or tuple
        if isinstance(f, list) or isinstance(f, tuple):
            self._f = f
        else:
            self._f = [f]

        # generate casADi function objects
        self._function = ca.Function("f", self.x, self._f)

        # generate indices
        # self._idx_x = [y.shape for y in self.x]
        self._idx_function = [y.shape for y in self._f]
        self._idx_x = self._idx_function[: len(self.x)]

    def function(self, x, threads=cpu_count()):
        "Return the function."
        return apply(
            x,
            fun=self._function,
            x_shape=self._idx_x,
            fun_shape=self._idx_function,
            threads=threads,
        )


class Material(Function):
    def __init__(self, x, fun, args=(), kwargs={}, compress=False):

        # init Function
        super().__init__(x=x, fun=fun, args=args, kwargs=kwargs)

        _h_diag = []
        _hvp_diag = []

        self._h = []
        self._g = []

        self._hvp = []
        self._gvp = []

        # generate vectors for gradient- and hessian-vector products
        self.v = [Variable("v%d" % a, *x.shape) for a, x in enumerate(self.x)]
        self.u = [Variable("u%d" % a, *x.shape) for a, x in enumerate(self.x)]

        # alias
        self.jacobian = self.gradient

        # generate list of diagonal hessian entries and gradients
        # (including vector-products)
        for x, v, u in zip(self.x, self.v, self.u):
            _h, _g = ca.hessian(self._f[0], x)
            _h_diag.append(_h)
            self._g.append(_g)

            _gvp = ca.jtimes(self._f[0], x, v)
            _hvp = ca.jtimes(_gvp, x, u)
            self._gvp.append(_gvp)
            _hvp_diag.append(_hvp)

        # generate upper-triangle of hessian (-vector-products)
        for i, (g, gvp) in enumerate(zip(self._g, self._gvp)):
            for j, (x, u) in enumerate(zip(self.x, self.u)):
                if j >= i:
                    if i != j:
                        self._h.append(ca.jacobian(g, x))
                        self._hvp.append(ca.jtimes(gvp, x, u))
                    else:
                        self._h.append(_h_diag[i])
                        self._hvp.append(_hvp_diag[i])

        # generate casADi function objects
        self._gradient = ca.Function("g", self.x, self._g)
        self._hessian = ca.Function("h", self.x, self._h)
        self._gradient_vector_product = ca.Function(
            "gvp", [*self.x, *self.v], self._gvp
        )
        self._hessian_vector_product = ca.Function(
            "hvp", [*self.x, *self.v, *self.u], self._hvp
        )

        # generate indices
        self._idx_hessian = []

        if compress:
            for i in range(len(self._idx_gradient)):
                if np.all(np.array(self._idx_gradient[i]) == 1):
                    self._idx_gradient[i] = ()

        for i in range(len(self._idx_gradient)):
            a = self._idx_gradient[i]

            for j in range(len(self._idx_gradient)):
                b = self._idx_gradient[j]

                if j >= i:
                    self._idx_hessian.append((*a, *b))

    def gradient(self, x, threads=cpu_count()):
        "Return list of gradients."
        return apply(
            x,
            fun=self._gradient,
            x_shape=self._idx_gradient,
            fun_shape=self._idx_gradient,
            threads=threads,
        )

    def hessian(self, x, threads=cpu_count()):
        "Return upper-triangle entries of hessian."
        return apply(
            x,
            fun=self._hessian,
            x_shape=self._idx_gradient,
            fun_shape=self._idx_hessian,
            threads=threads,
        )

    def gradient_vector_product(self, x, v, threads=cpu_count()):
        "Return list of gradient-vector-products."
        return apply(
            [*x, *v],
            fun=self._gradient_vector_product,
            x_shape=self._idx_gradient,
            fun_shape=self._idx_function * len(self._gvp),
            threads=threads,
        )

    def hessian_vector_product(self, x, v, u, threads=cpu_count()):
        "Return list of hessian-vector-products."
        return apply(
            [*x, *v, *u],
            fun=self._hessian_vector_product,
            x_shape=self._idx_gradient,
            fun_shape=self._idx_function * len(self._hvp),
            threads=threads,
        )


class MaterialTensor(FunctionTensor):
    def __init__(
        self, x, fun, args=(), kwargs={}, compress=False, triu=True, statevars=0
    ):

        # init Function
        super().__init__(x=x, fun=fun, args=args, kwargs=kwargs)

        # no. of active variables
        n = len(self.x) - statevars

        # generate vector for gradient-vector-product
        self.v = [Variable("v%d" % a, *x.shape) for a, x in enumerate(self.x)]

        # generate gradient and gradient-vector-product
        self._g = [ca.jacobian(f, x) for x in self.x[:n] for f in self._f[:n]]
        self._gvp = [
            ca.jtimes(f, x, v)
            for x, v in zip(self.x[:n], self.v[:n])
            for f in self._f[:n]
        ]

        # alias
        self.jacobian = self.gradient

        # store only upper-triangle entries of gradients
        if triu:
            i, j = np.triu_indices(len(self.x[:n]))
            a = (
                np.arange(len(self.x[:n]) ** 2)
                .reshape(len(self.x[:n]), len(self.x[:n]))[i, j]
                .ravel()
            )
            self._g = [self._g[b] for b in a]
            self._gvp = [self._gvp[b] for b in a]

        # generate casADi function objects
        self._function = ca.Function("f", self.x, self._f)
        self._gradient = ca.Function("g", self.x, self._g)
        self._gradient_vector_product = ca.Function(
            "gvp", [*self.x, *self.v], self._gvp
        )

        # generate indices
        self._idx_gradient = []

        if compress:
            for i in range(len(self._idx_function)):
                if np.all(np.array(self._idx_function[i]) == 1):
                    self._idx_x[i] = ()
                    self._idx_function[i] = ()

        for i in range(len(self._idx_function[:n])):
            a = self._idx_function[i]

            for j in range(len(self._idx_function[:n])):
                b = self._idx_function[j]

                if triu:
                    if j >= i:
                        self._idx_gradient.append((*a, *b))
                else:
                    self._idx_gradient.append((*a, *b))

    def gradient(self, x, threads=cpu_count()):
        "Return list of gradients."
        return apply(
            x,
            fun=self._gradient,
            x_shape=self._idx_x,
            fun_shape=self._idx_gradient,
            threads=threads,
        )

    def gradient_vector_product(self, x, threads=cpu_count()):
        "Return list of gradient-vector-products."
        return apply(
            x,
            fun=self._gradient_vector_product,
            x_shape=self._idx_x,
            fun_shape=self._idx_function * len(self._gvp),
            threads=threads,
        )
