import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.optimize import root


class Lab:
    def __init__(self, material):

        self.material = material
        self.title = self._get_title()

    def _get_title(self):
        return "Material Formulation: " + "-".join(
            [m.title() for m in self.material.fun.__name__.split("_")]
        )

    def _uniaxial(self, stretch):
        def stress(stretch, stretch_2, stretch_3):
            F = np.diag([stretch, stretch_2, stretch_3])
            return self.material.gradient([F])[0]

        def stability(stretch, stretches_23, stretches_23_eps):
            F = np.diag([stretch, *stretches_23])
            G = np.diag([stretch + 1e-6, *stretches_23_eps])
            A = self.material.hessian([F])[0]

            # convert hessian to (3, 3) matrix
            B = np.zeros((3, 3))
            c = [(0, 0), (1, 1), (2, 2)]

            for i, a in enumerate(c):
                for j, b in enumerate(c):
                    B[i, j] = A[(*a, *b)]

            # init unit force in direction 1
            df = np.zeros(3)
            df[0] = 1

            # calculate linear solution of stretch 1 resulting from unit load
            dl = (np.linalg.inv(B) @ df)[0]

            # check volume ratio
            J = stretch * stretches_23[0] * stretches_23[1]

            # check slope of force
            Q = self.material.gradient([G])[0][0, 0]
            P = self.material.gradient([F])[0][0, 0]

            return True if dl > 0 and J > 0 and (Q - P) > 0 else False

        def stress_free(stretches_23, stretch):
            s = stress(stretch, *stretches_23)
            return [s[1, 1], s[2, 2]]

        res = root(stress_free, np.ones(2), args=(stretch,))
        if not res.success:
            res = root(stress_free, np.ones(2) * 1 / np.sqrt(stretch), args=(stretch,))

        res_eps = root(stress_free, np.ones(2), args=(stretch + 1e-6,))
        if not res_eps.success:
            res_eps = root(
                stress_free,
                np.ones(2) * 1 / np.sqrt(stretch + 1e-6),
                args=(stretch + 1e-6,),
            )

        return (
            stress(stretch, *res.x)[0, 0],
            *res.x,
            stability(stretch, res.x, res_eps.x),
        )

    def _biaxial(self, stretch):
        def stress(stretch, stretch_3):
            F = np.diag([stretch, stretch, stretch_3])
            return self.material.gradient([F])[0]

        def stability(stretch, stretch_3, stretch_3_eps):
            F = np.diag([stretch, stretch, stretch_3])
            G = np.diag([stretch + 1e-6, stretch + 1e-6, stretch_3_eps])
            A = self.material.hessian([F])[0]

            # convert hessian to (3, 3) matrix
            B = np.zeros((3, 3))
            c = [(0, 0), (1, 1), (2, 2)]

            for i, a in enumerate(c):
                for j, b in enumerate(c):
                    B[i, j] = A[(*a, *b)]

            # init unit force in direction 1
            df = np.zeros(3)
            df[0] = 1

            # calculate linear solution of stretch 1 resulting from unit load
            dl = (np.linalg.inv(B) @ df)[0]

            # check volume ratio
            J = stretch ** 2 * stretch_3

            # check slope of force
            Q = self.material.gradient([G])[0][0, 0]
            P = self.material.gradient([F])[0][0, 0]

            return True if dl > 0 and J > 0 and (Q - P) > 0 else False

        def stress_free(stretch_3, stretch):
            return [stress(stretch, *stretch_3)[2, 2]]

        res = root(stress_free, np.ones(1), args=(stretch,))
        stretch_3 = res.x[0]

        res_eps = root(stress_free, np.ones(1), args=(stretch + 1e-6,))
        stretch_3_eps = res_eps.x[0]

        return (
            stress(stretch, stretch_3)[0, 0],
            stretch,
            stretch_3,
            stability(stretch, stretch_3, stretch_3_eps),
        )

    def _planar(self, stretch):
        def stress(stretch, stretch_3):
            F = np.diag([stretch, 1, stretch_3])
            return self.material.gradient([F])[0]

        def stability(stretch, stretch_3, stretch_3_eps):
            F = np.diag([stretch, 1, stretch_3])
            G = np.diag([stretch + 1e-6, 1, stretch_3_eps])
            A = self.material.hessian([F])[0]

            # convert hessian to (3, 3) matrix
            B = np.zeros((3, 3))
            c = [(0, 0), (1, 1), (2, 2)]

            for i, a in enumerate(c):
                for j, b in enumerate(c):
                    B[i, j] = A[(*a, *b)]

            # init unit force in direction 1
            df = np.zeros(3)
            df[0] = 1

            # calculate linear solution of stretch 1 resulting from unit load
            dl = (np.linalg.inv(B) @ df)[0]

            # check volume ratio
            J = stretch * stretch_3

            # check slope of force
            Q = self.material.gradient([G])[0][0, 0]
            P = self.material.gradient([F])[0][0, 0]

            return True if dl > 0 and J > 0 and (Q - P) > 0 else False

        def stress_free(stretch_3, stretch):
            return [stress(stretch, *stretch_3)[2, 2]]

        res = root(stress_free, np.ones(1), args=(stretch,))
        stretch_3 = res.x[0]

        res_eps = root(stress_free, np.ones(1), args=(stretch + 1e-6,))
        stretch_3_eps = res_eps.x[0]

        return (
            stress(stretch, stretch_3)[0, 0],
            1,
            stretch_3,
            stability(stretch, stretch_3, stretch_3_eps),
        )

    def _shear(self, shear):
        def stress(shear, stretch_2, stretch_3):
            F = np.diag([1, stretch_2, stretch_3])
            F[0, 1] = shear
            return self.material.gradient([F])[0]

        def stress_free(stretches, shear):
            return [stress(shear, *stretches)[1, 1], stress(shear, *stretches)[2, 2]]

        res = root(stress_free, np.ones(2), args=(shear,))
        stretches = res.x

        return (
            stress(shear, *stretches)[1, 0],
            stretches[0],
            stretches[1],
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

        out = []

        Data = namedtuple(
            "Data", "label stretch stretch_2 stretch_3 shear stress stability"
        )

        if ux:
            if stretch_min is None:
                stretch_min = max(0, 1 - (stretch_max - 1) / 5)

            stretch_ux = np.linspace(stretch_min, stretch_max, num)
            stress_ux, stretch_2_ux, stretch_3_ux, stability_ux = np.array(
                [self._uniaxial(s11) for s11 in stretch_ux]
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
                [self._biaxial(s11) for s11 in stretch_bx]
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
                [self._planar(s11) for s11 in stretch_ps]
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
                stretch_2_sh,
                stretch_3_sh,
                deformation_sh,
                stress_shear,
                stability_sh,
            )
            out.append(shear_data)

        return out

    def plot(self, data, stability=False):

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
        ax.set_ylabel(r"force per undeformed area $P_{21} \quad \longrightarrow$")
        ax.legend()

        fig.tight_layout()

        return fig, ax
