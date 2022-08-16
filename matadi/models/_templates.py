from ._misc import morph
from ._hyperelasticity_isotropic import neo_hooke
from ._pseudo_elasticity import ogden_roxburgh
from ._helpers import volumetric, displacement_pressure_split
from ..math import det, gradient
from .._templates import MaterialTensorGeneral


class NeoHookeOgdenRoxburgh(MaterialTensorGeneral):
    "Neo-Hooke and Ogden-Roxburgh material formulations within the u/p framework."

    def __init__(self, C10=0.5, r=3, m=1, beta=0, bulk=5000):
        @displacement_pressure_split
        def fun(x, C10, r, m, beta, bulk):

            # split `x` into the deformation gradient and the state variable
            F, Wmaxn = x[0], x[-1]

            # isochoric and volumetric parts of the hyperelastic
            # strain energy function
            W = neo_hooke(F, C10)
            U = volumetric(det(F), bulk)

            # pseudo-elastic softening function
            eta, Wmax = ogden_roxburgh(W, Wmaxn, r, m, beta)

            # first Piola-Kirchhoff stress and updated state variable
            # for a pseudo-elastic material formulation
            return eta * gradient(W, F) + gradient(U, F), Wmax

        super().__init__(fun=fun, statevars_shape=(1, 1), C10=C10, r=r, m=m, beta=beta, bulk=bulk)


class Morph(MaterialTensorGeneral):
    "MORPH consitutive material formulation within the u/p framework."

    def __init__(
        self,
        p1=0.035,
        p2=0.37,
        p3=0.17,
        p4=2.4,
        p5=0.01,
        p6=6.4,
        p7=5.5,
        p8=0.24,
        bulk=4050,
    ):
        @displacement_pressure_split
        def fun(x, p1, p2, p3, p4, p5, p6, p7, p8, bulk):

            F = x[0]
            J = det(F)

            P, statevars = morph(x, p1, p2, p3, p4, p5, p6, p7, p8)
            U = volumetric(J, bulk)

            return P + gradient(U, F), statevars

        super().__init__(
            fun=fun,
            statevars_shape=(13, 1),
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5=p5,
            p6=p6,
            p7=p7,
            p8=p8,
            bulk=bulk,
        )
