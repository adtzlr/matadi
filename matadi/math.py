from casadi import (
    # tensor and vector operations
    det,
    inv,
    dot as _dot,
    transpose,
    trace,
    diag,
    triu,
    tril,
    adj,
    cofactor,
    cross,
    times,
    eig_symbolic,
    qr,
    ldl,
    # trig
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    asin,
    acos,
    atan,
    atan2,
    asinh,
    acosh,
    atanh,
    # math
    exp,
    log,
    sqrt,
    sum1,
    sum2,
    sumsqr,
    pi,
    fabs,
    linspace,
    erf,
    erfinv,
    norm_1,
    sign,
    fmin,
    fmax,
    mmin,
    mmax,
    find,
    #
    if_else,
    logic_and,
    logic_or,
    logic_not,
    floor,
    ceil,
    SX,
    DM,
    MX,
    #
    vertcat,
    horzcat,
    vertsplit,
    horzsplit,
    repmat,
    reshape,
    gradient,
    hessian,
    Function,
)

eye = SX.eye
ones = SX.ones
zeros = SX.zeros


def zeros_like(T):

    return zeros(T.shape)


def ones_like(T):

    return ones(T.shape)


def invariants(T):

    I1 = trace(T)
    I2 = (I1**2 - trace(T @ T)) / 2
    I3 = det(T)

    return I1, I2, I3


def eigvals(T, eps=1e-4):

    D = DM([[1, 0, 0], [0, -1, 0], [0, 0, 0]])

    return eig_symbolic(T + D * eps)


def cof(T):

    return det(T) * transpose(inv(T))


def sym(T):

    return (T + transpose(T)) / 2


def dot(A, B):

    return _dot(transpose(A), B)


def dev(T):

    dim = T.shape[0]

    return T - trace(T) / dim * eye(dim)


def ddot(A, B):

    return trace(transpose(A) @ B)


def tresca(C):
    "Tresca Invariant as maximum difference of two eigenvalues."
    wC = eigvals(C, 8e-5)
    return mmax(fabs(wC[[0, 1, 2]] - wC[[1, 2, 0]]))


def mexp(C, eps=8e-5):
    "Exponential Function of a Matrix."
    w = eigvals(C, eps=eps)
    I = SX.eye(3)

    M1 = (C - w[1] * I) * (C - w[2] * I) / (w[0] - w[1]) / (w[0] - w[2])
    M2 = (C - w[2] * I) * (C - w[0] * I) / (w[1] - w[2]) / (w[1] - w[0])
    M3 = (C - w[0] * I) * (C - w[1] * I) / (w[2] - w[0]) / (w[2] - w[1])

    return exp(w[0]) * M1 + exp(w[1]) * M2 + exp(w[2]) * M3


def asvoigt(A, scale=1):

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
