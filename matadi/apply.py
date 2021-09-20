from functools import partial
import numpy as np
import casadi as ca


def apply(x, fun, fun_shape, trailing_axes=2):
    "Helper function for the calculation of fun(x)."

    # get shape of trailing axes
    ax = x[0].shape[-trailing_axes:]

    def rshape(z):
        "Reshape array `z`: 'i,j,...->i,...'."
        if len(z.shape) == trailing_axes:
            return z.reshape(1, -1, order="F")
        else:
            return z.reshape(z.shape[0], -1, order="F")

    # apply reshape on input
    y = [rshape(z) for z in x]

    # map function `N` times on reshaped input
    N = np.product(ax)
    out = np.array(fun.map(N)(*y))

    # return 'i,j,...' reshaped output
    return out.reshape(*fun_shape, *ax, order="F")


class FunctionScalar:
    def __init__(self, x, fun, *args, **kwargs):
        
        self.x = x
        self.fun = fun
        
        self.args = args
        self.kwargs = kwargs
        
        self._f = self.fun(self.x, *self.args, **self.kwargs)
        
        self._h = []
        self._g = []
        
        for y in self.x:
            _h, _g = ca.hessian(self._f, y)
            self._h.append(_h)
            self._g.append(_g)
        
        # generate casADi function objects
        self._grad = ca.Function("g", self.x, self._g)
        self._hessian = ca.Function("h", self.x, self._h)
    
    def grad(self, x):
        ij = x[0].shape[:2]
        return apply(x, fun=self._grad, fun_shape=ij)
    
    def hessian(self, x):
        ij = x[0].shape[:2]
        return apply(x, fun=self._hessian, fun_shape=(*ij, *ij))

x = ca.SX.sym("x", 3, 3)
f = lambda x: ca.trace(x[0])

W = FunctionScalar([x], f)

y = np.random.rand(3,3,5,100)
W.grad(y)