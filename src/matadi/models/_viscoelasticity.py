from ..math import astensor, asvoigt, det, eye, gradient, inv, sqrtm, trace, unimodular


def finite_strain_viscoelastic(x, mu, eta, dtime):
    "Finite strain viscoelastic material formulation."

    # split input into the deformation gradient and the vector of state variables
    F, Cin = x[0], x[-1]

    # update of state variables by evolution equation
    Ci = astensor(Cin) + mu / eta * dtime * det(F) ** (-2 / 3) * (F.T @ F)
    Ci = det(Ci) ** (-1 / 3) * Ci

    # first invariant of elastic part of right Cauchy-Green deformation tensor
    I1 = det(F) ** (-2 / 3) * trace((F.T @ F) @ inv(Ci))

    # first Piola-Kirchhoff stress tensor and state variable
    return gradient(mu / 2 * (I1 - 3), F), asvoigt(Ci)


def finite_strain_viscoelastic_mr(x, c10, c01, eta, dtime):
    """
    Finite strain viscoelastic material formulation with Mooney-Rivlin hyperelasticity
    (Shutov 2018) https://doi.org/10.1002/nme.5724
    """

    # Split input into the deformation gradient and the vector of state variables
    F, Cin = x[0], x[-1]

    # Right cauchy-green deformation tensor
    C = F.T @ F

    # Based on
    # <<TABLE 1: Iteration-free Euler backward method on the reference configuration>>
    A = (
        unimodular(sqrtm(inv(C)))
        @ (astensor(Cin) + (dtime / eta) * c10 * unimodular(C))
        @ unimodular(sqrtm(inv(C)))
    )
    eps = c01 * (dtime / eta)
    phi0 = det(A) ** (1 / 3)
    phi = phi0 - (trace(A) / (3 * phi0)) * eps
    X = 2 * A @ inv(sqrtm(phi * phi * eye(3) + 4 * eps * A) + phi * eye(3))

    Ci = unimodular(sqrtm(C) @ X @ sqrtm(C))

    I1 = trace(unimodular(C @ inv(Ci)))
    I2 = trace(unimodular(Ci @ inv(C)))

    # First Piola-Kirchhoff stress tensor and state variable
    return gradient(c10 / 2 * (I1 - 3) + c01 / 2 * (I2 - 3), F), asvoigt(Ci)
