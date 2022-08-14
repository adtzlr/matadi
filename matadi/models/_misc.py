from ._helpers import displacement_pressure_split, volumetric
from ._hyperelasticity_isotropic import neo_hooke
from ..math import (
    det, inv, dev, sym, sqrt, if_else, vertsplit, vertcat, 
    asvoigt, astensor, tresca, mexp, gradient
)


@displacement_pressure_split
def morph(x, param, bulk):
    "MORPH consitututive material formulation."
    
    # split input into the deformation gradient and the vector of state variables
    F, statevars = x[0], x[-1]
    
    # split state variables
    CTSn, Cn, SZn = vertsplit(statevars, [0, 1, 7, 13])
    Cn, SZn = astensor(Cn), astensor(SZn)
    
    # right Cauchy-Green deformation tensor
    C = F.T @ F
    J = det(F)
    dC = C - Cn
    
    # (Incremental) Inverse  of right Cauchy-Green deformation tensor
    invC = inv(C)
    
    # isochoric part of C and L
    CG = J ** (-2 / 3) * C
    LG = dev(sym(dC @ invC)) @ CG
    
    # # von mises invariants of C and L
    CT = tresca(CG)
    LT = tresca(LG)
    
    # # maximum historical von mises invariant of CG
    CTS = if_else(CT > CTSn, CT, CTSn)
    
    # # stable LG / LT and CT / CTS
    LGLT = if_else(LT > 0, LG / LT, LG)
    CTCTS = if_else(CTS > 0, CT / CTS, CT)
    
    # MORPH deformation-dependent material parameters
    f = lambda x: 1 / sqrt(1 + x ** 2)
    a = param[0] + param[1] * f(param[2] * CTS)
    b = param[3] * f(param[2] * CTS)
    c = param[4] * CTS * (1 - f(CTS / param[5]))
    
    # # Hull stress
    SH = (c * mexp(param[6] * LGLT * CTCTS) + param[7] * LGLT) @ invC
    
    # # helper "kappa" for implict euler update of overstress
    SZ = (SZn + b * LT * SH) / (1 + b * LT)

    # hyperelastic distortional and volumetric parts of strain energy function
    W = neo_hooke(F, a)
    U = volumetric(J, bulk)
    
    # overstress as second Piola-Kirchhoff stress
    S = dev(SZ @ C) @ invC
    
    # update statevars
    statevars_new = vertcat(CTS, asvoigt(C), asvoigt(SZ))
    
    # first Piola-Kirchhoff stress
    return [gradient(W, F) + gradient(U, F) + F @ S, statevars_new]