from casadi import (
    # tensor and vector operations
    det,
    inv,
    dot,
    transpose,
    trace,
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
    #
    if_else,
    SX,
    DM,
    MX,
)


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
