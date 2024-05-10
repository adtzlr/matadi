from casadi import (
    DM,
    MX,
    SX,
    Function,
    acos,
    acosh,
    adj,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    ceil,
    cofactor,
    cos,
    cosh,
    cross,
    det,
    diag,
)
from casadi import dot as _dot  # tensor and vector operations; trig; math
from casadi import (
    eig_symbolic,
    erf,
    erfinv,
    exp,
    fabs,
    find,
    floor,
    fmax,
    fmin,
    gradient,
    hessian,
    horzcat,
    horzsplit,
    if_else,
    inv,
    ldl,
    linspace,
    log,
    logic_and,
    logic_not,
    logic_or,
    mmax,
    mmin,
    norm_1,
    pi,
    qr,
    repmat,
    reshape,
    sign,
    sin,
    sinh,
    sqrt,
    sum1,
    sum2,
    sumsqr,
    tan,
    tanh,
    times,
    trace,
    transpose,
    tril,
    triu,
    vertcat,
    vertsplit,
)

__all__ = [
    "DM",
    "MX",
    "SX",
    "Function",
    "acos",
    "acosh",
    "adj",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "ceil",
    "cofactor",
    "cos",
    "cosh",
    "cross",
    "det",
    "diag",
    "_dot",
    "eig_symbolic",
    "erf",
    "erfinv",
    "exp",
    "fabs",
    "find",
    "floor",
    "fmax",
    "fmin",
    "gradient",
    "hessian",
    "horzcat",
    "horzsplit",
    "if_else",
    "inv",
    "ldl",
    "linspace",
    "log",
    "logic_and",
    "logic_not",
    "logic_or",
    "mmax",
    "mmin",
    "norm_1",
    "pi",
    "qr",
    "repmat",
    "reshape",
    "sign",
    "sin",
    "sinh",
    "sqrt",
    "sum1",
    "sum2",
    "sumsqr",
    "tan",
    "tanh",
    "times",
    "trace",
    "transpose",
    "tril",
    "triu",
    "vertcat",
    "vertsplit",
]

eye = SX.eye
ones = SX.ones
zeros = SX.zeros


def zeros_like(T):
    "Return an array of zeros with the same shape and type as a given array."
    return zeros(T.shape)


def ones_like(T):
    "Return an array of ones with the same shape and type as a given array."
    return ones(T.shape)


def invariants(T):
    "Return the three principal invariants."

    I1 = trace(T)
    I2 = (I1**2 - trace(T @ T)) / 2
    I3 = det(T)

    return I1, I2, I3


def eigvals(T, eps=1e-4):
    """Compute the eigenvalues of a 3x3 matrix, perturbed by a small number ``eps`` on
    the diagonal entries."""

    # perturbation matrix
    D = DM([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

    return eig_symbolic(T + D * eps)


def cof(T):
    "Return the cofactor matrix."
    return det(T) * transpose(inv(T))


def sym(T):
    "Return the symmetric part of an array."
    return (T + transpose(T)) / 2


def dot(A, B):
    "Return the single-contraction (dot-) product (matrix multiplication)."
    return _dot(transpose(A), B)


def dev(T):
    "Return the deviatoric part of a matrix."
    dim = T.shape[0]
    return T - trace(T) / dim * eye(dim)


def ddot(A, B):
    "Return the double-dot product (the sum of all element-wise products)."
    return trace(transpose(A) @ B)


def tresca(C):
    "Tresca Invariant as maximum difference of two eigenvalues."
    wC = eigvals(C, 8e-5)
    return mmax(fabs(wC[[0, 1, 2]] - wC[[1, 2, 0]]))


def mexp(C, eps=8e-5):
    "Exponential Function of a Matrix."
    w = eigvals(C, eps=eps)
    eye = SX.eye(3)

    M1 = (C - w[1] * eye) * (C - w[2] * eye) / (w[0] - w[1]) / (w[0] - w[2])
    M2 = (C - w[2] * eye) * (C - w[0] * eye) / (w[1] - w[2]) / (w[1] - w[0])
    M3 = (C - w[0] * eye) * (C - w[1] * eye) / (w[2] - w[0]) / (w[2] - w[1])

    return exp(w[0]) * M1 + exp(w[1]) * M2 + exp(w[2]) * M3


def asvoigt(A, scale=1):
    """Return a 2x2 or 3x3 symmetric matrix in 3x1 or 6x1 reduced vector (Voigt)
    storage. Only the upper-triangle part of a given matrix is considered. Optionally,
    the off-diagonal items are scaled by a given scale factor.
    """
    if A.shape == (3, 3):
        return vertcat(
            A[0, 0],
            A[1, 1],
            A[2, 2],
            A[0, 1] * scale,
            A[1, 2] * scale,
            A[0, 2] * scale,
        )

    elif A.shape == (2, 2):
        return vertcat(
            A[0, 0],
            A[1, 1],
            A[0, 1] * scale,
        )

    else:
        raise ValueError("Unknown shape of input.")


def astensor(A, scale=1):
    """Return a 3x1 or 6x1 vector, which represents a symmetric matrix, in 2x2 or 3x3
    full matrix storage. Optionally, the off-diagonal items are scaled by a given scale
    factor.
    """
    if A.shape == (6, 1):
        A0 = vertcat(A[0] / scale, A[3] / scale, A[5])
        A1 = vertcat(A[3] / scale, A[1], A[4] / scale)
        A2 = vertcat(A[5] / scale, A[4] / scale, A[2])
        return horzcat(A0, A1, A2)

    elif A.shape == (3, 1):
        A0 = vertcat(A[0], A[2] / scale)
        A1 = vertcat(A[2] / scale, A[1])
        return horzcat(A0, A1)

    else:
        raise ValueError("Unknown shape of input.")

def unimodular(T):
   """
   Compute the unimodular part of a tensor.

   The unimodular part of a tensor is a modified version of the tensor where
   the determinant is raised to the power of (-1/3) and multiplied to the tensor.
   This operation preserves the isochoric (volume-preserving) part of the tensor
   while removing the volumetric part.
   """
   return (det(T) ** (-1 / 3)) * T

def sqrtm(C, eps = 8e-5):
    """
    Compute the matrix square root of a tensor C using eigendecomposition.
    """
    w = eigvals(C, eps=eps)
    eye = SX.eye(3)

    M1 = (C - w[1] * eye) * (C - w[2] * eye) / (w[0] - w[1]) / (w[0] - w[2])
    M2 = (C - w[2] * eye) * (C - w[0] * eye) / (w[1] - w[2]) / (w[1] - w[0])
    M3 = (C - w[0] * eye) * (C - w[1] * eye) / (w[2] - w[0]) / (w[2] - w[1])

    return sqrt(w[0]) * M1 + sqrt(w[1]) * M2 + sqrt(w[2]) * M3