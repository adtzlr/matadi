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


class Scalar:
    def __init__(self, fun, *args, **kwargs):
        
        self.x = ca.SX.sym("x", 3, 3)
        self.fun = fun
        
        self.args = args
        self.kwargs = kwargs
        
        f = self.fun(self.x, *self.args, **self.kwargs)
        
        hessian, gradient = ca.hessian(f, x)