import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.optimize import root


class LabIncompressible:
    def __init__(self, material):

        self.material = material
        self.title = self._get_title()

    def _get_title(self):
        return "Material Formulation: " + "-".join(
            [m.title() for m in self.material.fun.__name__.split("_")]
        )

    def _ux(self, stretch):
        "Principal stretches of incompressible uniaxial load case."
        return stretch, 1 / np.sqrt(stretch), 1 / np.sqrt(stretch)

    def _bx(self, stretch):
        "Principal stretches of incompressible equi-biaxial load case."
        return stretch, stretch, 1 / stretch ** 2

    def _ps(self, stretch):
        "Principal stretches of incompressible planar shear load case."
        return stretch, 1, 1 / stretch

    def _loadcase(self, stretch, kinematics):
        "Generalized load case `UX/BX/PS (Incompressible)`."

        def stress(stretch):
            "Evaluate the first Piola-Kirchhoff stress tensor."
            F = np.diag([*kinematics(stretch)])
            return self.material.gradient([F])[0]

        def stress_free(stretch):
            """First Piola-Kirchhoff stress tensor with resolved hydrostatic
            pressure due to incompressibility."""
            P = stress(stretch)
            return P[0, 0] - kinematics(stretch)[-1] / stretch * P[-1, -1]

        def stability(stretch):
            F = np.diag([*kinematics(stretch)])
            G = np.diag([*kinematics(stretch + 1e-6)])

            P = self.material.gradient([F])[0]
            A = self.material.hessian([F])[0]

            # convert hessian to (3, 3) matrix and take (2, 2) submatrix
            B = np.zeros((2, 2))
            delta = np.eye(2)

            for i in range(2):
                for j in range(2):
                    B[i, j] = (
                        A[i, i, j, j]
                        - delta[-1, j] / kinematics(stretch)[i] * P[-1, -1]
                        - kinematics(stretch)[-1]
                        / kinematics(stretch)[i]
                        * A[-1, -1, j, j]
                        + kinematics(stretch)[-1]
                        / kinematics(stretch)[i] ** 2
                        * P[-1, -1]
                        * delta[i, j]
                    )

            # init unit force in direction 1
            df = np.zeros(2)
            df[0] = 1

            # calculate linear solution of stretch 1 resulting from unit load
            dl = (np.linalg.inv(B) @ df)[0]

            # check slope of force
            Q = self.material.gradient([G])[0][0, 0]
            P = self.material.gradient([F])[0][0, 0]

            return True if dl > 0 and (Q - P) > 0 else False

        return (
            stress_free(stretch),
            *kinematics(stretch)[1:],
            stability(stretch),
        )

    def _shear(self, shear):
        "Load case `Simple Shear (Incompressible)`."

        def stress(shear):
            F = np.eye(3)
            F[0, 1] = shear
            return self.material.gradient([F])[0]

        return (
            stress(shear)[0, 1],
            1,
            1,
            None,
        )

    def run(
        self,
        ux=True,
        bx=True,
        ps=True,
        shear=True,
        stretch_min=None,
        stretch_max=2.5,
        shear_max=1.0,
        num=50,
    ):
        "Run load cases `UX/BX/PS/Simple-Shear (Incompressible)`."

        out = []

        Data = namedtuple(
            "Data", "label stretch stretch_2 stretch_3 shear stress stability"
        )

        if ux:
            if stretch_min is None:
                stretch_min = max(0, 1 - (stretch_max - 1) / 5)

            stretch_ux = np.linspace(stretch_min, stretch_max, num)
            stress_ux, stretch_2_ux, stretch_3_ux, stability_ux = np.array(
                [self._loadcase(s11, kinematics=self._ux) for s11 in stretch_ux]
            ).T
            ux_data = Data(
                "uniaxial",
                stretch_ux,
                stretch_2_ux,
                stretch_3_ux,
                0,
                stress_ux,
                stability_ux,
            )
            out.append(ux_data)

        if bx:
            stretch_bx = np.linspace(1, (stretch_max - 1) / 2 + 1, num)
            stress_bx, stretch_2_bx, stretch_3_bx, stability_bx = np.array(
                [self._loadcase(s11, kinematics=self._bx) for s11 in stretch_bx]
            ).T
            bx_data = Data(
                "biaxial",
                stretch_bx,
                stretch_2_bx,
                stretch_3_bx,
                0,
                stress_bx,
                stability_bx,
            )
            out.append(bx_data)

        if ps:
            stretch_ps = np.linspace(1, stretch_max, num)
            stress_ps, stretch_2_ps, stretch_3_ps, stability_ps = np.array(
                [self._loadcase(s11, kinematics=self._ps) for s11 in stretch_ps]
            ).T
            ps_data = Data(
                "planar",
                stretch_ps,
                stretch_2_ps,
                stretch_3_ps,
                0,
                stress_ps,
                stability_ps,
            )
            out.append(ps_data)

        if shear:
            deformation_sh = np.linspace(0, shear_max, num)
            stress_shear, stretch_2_sh, stretch_3_sh, stability_sh = np.array(
                [self._shear(g) for g in deformation_sh]
            ).T
            shear_data = Data(
                "shear",
                1,
                1,
                1,
                deformation_sh,
                stress_shear,
                stability_sh,
            )
            out.append(shear_data)

        return out

    def plot(self, data, stability=False):
        "Plot results of UX/BX/PS load cases."

        fig, ax = plt.subplots()

        lineargs = {
            "uniaxial": {"color": "C0"},
            "biaxial": {"color": "C1"},
            "planar": {"color": "C2"},
        }

        if data[-1].label == "shear":
            data = data[:-1]

        for d in data:

            if stability:

                stable = np.array(d.stability, dtype=bool)

                unstable_idx = np.arange(len(d.stretch))[~stable]
                stable_idx = np.arange(len(d.stretch))[stable]

                stress_stable = d.stress.copy()
                stress_stable[unstable_idx] = np.nan

                stress_unstable = d.stress.copy()
                stress_unstable[stable_idx] = np.nan

                ax.plot(d.stretch, stress_stable, **lineargs[d.label], label=d.label)
                ax.plot(
                    d.stretch,
                    stress_unstable,
                    **lineargs[d.label],
                    linestyle="--",
                )

            else:

                ax.plot(d.stretch, d.stress, **lineargs[d.label], label=d.label)

        ax.grid()
        ax.set_title(self.title, fontsize=10)
        ax.set_xlabel(r"stretch $\lambda_1 \quad \longrightarrow$")
        ax.set_ylabel(r"force per undeformed area $P_{11} \quad \longrightarrow$")
        ax.legend()

        fig.tight_layout()

        return fig, ax

    def plot_shear(self, data):
        "Plot results of shear load case."

        fig, ax = plt.subplots()

        lineargs = {
            "shear": {"color": "C3"},
        }

        if data[-1].label == "shear":
            d = data[-1]
        else:
            raise TypeError("No shear data found.")

        ax.plot(d.shear, d.stress, **lineargs[d.label], label=d.label)

        ax.grid()
        ax.set_title(self.title, fontsize=10)
        ax.set_xlabel(r"shear deformation $F_{12}=\gamma \quad \longrightarrow$")
        ax.set_ylabel(r"force per undeformed area $P_{12} \quad \longrightarrow$")
        ax.legend()

        fig.tight_layout()

        return fig, ax
