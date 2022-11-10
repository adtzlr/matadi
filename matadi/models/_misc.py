from ._hyperelasticity_isotropic import neo_hooke
from ..math import (
    det,
    inv,
    dev,
    sym,
    sqrt,
    if_else,
    vertsplit,
    vertcat,
    asvoigt,
    astensor,
    tresca,
    mexp,
    gradient,
)


def morph(x, p1, p2, p3, p4, p5, p6, p7, p8):
    "MORPH consitutive material formulation."

    # split input into the deformation gradient and the vector of state variables
    F, statevars = x[0], x[-1]

    # split state variables
    CTSn, Cn, SZn = vertsplit(statevars, [0, 1, 7, 13])
    Cn, SZn = astensor(Cn), astensor(SZn)

    # determimant of deformtion gradient
    J = det(F)

    # isochoric part of right Cauchy-Green deformation tensor
    C = F.T @ F
    CG = J ** (-2 / 3) * C

    # incremental right Cauchy-Green deformation tensor
    dC = C - Cn

    # (isochoric part of) lagrangian rate of deformation tensor
    L = dev(sym(dC @ inv(C))) @ CG

    # tresca invariants of (distortional part of) C and L
    CT = tresca(CG)
    LT = tresca(L)

    # maximum historical tresca invariant of (distortional part of) C
    CTS = if_else(CT > CTSn, CT, CTSn)

    # stable normalizations: L / LT and CT / CTS
    L_LT = if_else(LT > 0, L / LT, L)
    CT_CTS = if_else(CTS > 0, CT / CTS, CT)

    # MORPH deformation-dependent material parameters
    f = lambda x: 1 / sqrt(1 + x**2)
    a = p1 + p2 * f(p3 * CTS)
    b = p4 * f(p3 * CTS)
    c = p5 * CTS * (1 - f(CTS / p6))

    # Hull stress
    SH = (c * mexp(p7 * L_LT * CT_CTS) + p8 * L_LT) @ inv(C)

    # implict euler update of overstress evolution equation
    SZ = (SZn + b * LT * SH) / (1 + b * LT)

    # hyperelastic part of strain energy function
    W = neo_hooke(F, C10=0.5)

    # final overstress as first Piola-Kirchhoff stress
    PZ = F @ (dev(SZ @ C) @ inv(C))

    # update state variables
    statevars_new = vertcat(CTS, asvoigt(C), asvoigt(SZ))

    # total first Piola-Kirchhoff stress and new state variables
    return [2 * a * gradient(W, F) + PZ, statevars_new]
