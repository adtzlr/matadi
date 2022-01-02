from multiprocessing import cpu_count

import numpy as np
import casadi as ca

from ._apply import apply


class Function:
    def __init__(self, x, fun, args=(), kwargs={}, compress=False):

        self.x = x
        self._fun = fun

        self.args = args
        self.kwargs = kwargs

        # generate function
        self._f = self._fun(self.x, *self.args, **self.kwargs)

        # generate casADi function objects
        self._function = ca.Function("f", self.x, [self._f])

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
        self._f = self._fun(self.x, *self.args, **self.kwargs)

        # generate casADi function objects
        self._function = ca.Function("f", self.x, [self._f])

        # generate indices
        self._idx_function = [y.shape for y in x]

    def function(self, x, threads=cpu_count()):
        "Return the function."
        return apply(
            x,
            fun=self._function,
            x_shape=self._idx_function,
            fun_shape=self._idx_function,
            threads=threads,
        )


class Material(Function):
    def __init__(self, x, fun, args=(), kwargs={}, compress=False):

        # init Function
        super().__init__(x=x, fun=fun, args=args, kwargs=kwargs)

        _h_diag = []

        self._h = []
        self._g = []

        # alias
        self.jacobian = self.gradient

        # generate list of diagonal hessian entries and gradients
        for y in self.x:
            _hy, _gy = ca.hessian(self._f, y)
            _h_diag.append(_hy)
            self._g.append(_gy)

        # generate upper-triangle of hessian
        for i, g in enumerate(self._g):
            for j, y in enumerate(self.x):
                if j >= i:
                    if i != j:
                        self._h.append(ca.jacobian(g, y))
                    else:
                        self._h.append(_h_diag[i])

        # generate casADi function objects
        self._gradient = ca.Function("g", self.x, self._g)
        self._hessian = ca.Function("h", self.x, self._h)

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


class MaterialTensor(FunctionTensor):
    def __init__(self, x, fun, args=(), kwargs={}, compress=False):

        # init Function
        super().__init__(x=x, fun=fun, args=args, kwargs=kwargs)

        # generate gradients
        self._g = [ca.jacobian(self._f, y) for y in self.x]

        # alias
        self.jacobian = self.gradient

        # generate casADi function objects
        self._function = ca.Function("f", self.x, [self._f])
        self._gradient = ca.Function("g", self.x, self._g)

        # generate indices
        self._idx_gradient = []

        if compress:
            for i in range(len(self._idx_function)):
                if np.all(np.array(self._idx_function[i]) == 1):
                    self._idx_function[i] = ()

        for i in range(len(self._idx_function)):
            a = self._idx_function[i]

            for j in range(len(self._idx_function)):
                b = self._idx_function[j]

                self._idx_gradient.append((*a, *b))

    def gradient(self, x, threads=cpu_count()):
        "Return list of gradients."
        return apply(
            x,
            fun=self._gradient,
            x_shape=self._idx_function,
            fun_shape=self._idx_gradient,
            threads=threads,
        )
