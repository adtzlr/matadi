import numpy as np


def apply(x, fun, x_shape, fun_shape, threads=1):
    "Helper function for the calculation of fun(x)."

    # get shape of trailing axes
    trailing_axes = [len(y.shape) - len(y_shape) for y, y_shape in zip(x, x_shape)][0]

    if trailing_axes == 0:
        ax = [1]
    else:
        ax = x[0].shape[-trailing_axes:]

    def rshape(z):
        "Reshape array `z`: 'i,j,...->i,...'."
        if len(z.shape) == trailing_axes:
            return z.reshape(1, -1, order="F")
        else:
            return z.reshape(z.shape[0], -1, order="F")

    # apply reshape on input
    y = [rshape(z) for z in x]

    # threads dict
    if threads > 1:
        parallel = ("thread", threads)
    else:
        parallel = ()

    # map function `N` times on reshaped input
    N = np.product(ax)
    out = fun.map(N, *parallel)(*y)

    if not isinstance(out, tuple):
        out = (out,)

    # return 'i,j,...' reshaped output
    if trailing_axes == 0:
        ax = ()
        if fun_shape == [()]:
            fun_shape = [(1,)]

    return [np.array(o).reshape(*f, *ax, order="F") for o, f in zip(out, fun_shape)]
