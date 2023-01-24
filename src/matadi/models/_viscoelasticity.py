from ..math import astensor, asvoigt, det, gradient, inv, trace


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
