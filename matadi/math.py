from casadi import (
    # tensor and vector operations
    det,
    inv,
    dot as _dot,
    transpose,
    trace,
    diag,
    adj,
    cofactor,
    cross,
    times,
    eig_symbolic,
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
    reshape,
)

eye = SX.eye
ones = SX.ones
zeros = SX.zeros


def invariants(T):

    I1 = trace(T)
    I2 = (I1 ** 2 - trace(T @ T)) / 2
    I3 = det(T)

    return I1, I2, I3


def eigvals(T, eps=1e-5):

    T[0, 0] += eps
    T[1, 1] -= eps

    wT = eig_symbolic(T)

    return wT


def cof(T):

    return det(T) * transpose(inv(T))


def sym(T):

    return (T + transpose(T)) / 2


def dot(A, B):

    return _dot(transpose(A), B)
