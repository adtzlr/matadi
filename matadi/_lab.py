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
        def stress(stretch_2, stretch_3):
            F = np.diag([stretch, stretch_2, stretch_3])
            return self.material.gradient([F])[0]

        def stress_free(stretches_23):
            return [stress(*stretches_23)[1, 1], stress(*stretches_23)[2, 2]]

        res = root(stress_free, np.ones(2))
        stretch_2, stretch_3 = res.x

        return stress(stretch_2, stretch_3)[0, 0], stretch_2, stretch_3

    def _biaxial(self, stretch):
        def stress(stretch_3):
            F = np.diag([stretch, stretch, stretch_3])
            return self.material.gradient([F])[0]

        def stress_free(stretch_3):
            return [stress(*stretch_3)[2, 2]]

        res = root(stress_free, np.ones(1))
        stretch_3 = res.x[0]

        return stress(stretch_3)[0, 0], stretch, stretch_3

    def _planar(self, stretch):
        def stress(stretch_3):
            F = np.diag([stretch, 1, stretch_3])
            return self.material.gradient([F])[0]

        def stress_free(stretch_3):
            return [stress(*stretch_3)[2, 2]]

        res = root(stress_free, np.ones(1))
        stretch_3 = res.x[0]

        return stress(stretch_3)[0, 0], 1, stretch_3

    def run(self, ux=True, bx=True, ps=True, stretch_max=2.5, num=50):

        out = []

        Data = namedtuple("Data", "label stretch stretch_2 stretch_3 stress")

        if ux:
            stretch_ux = np.linspace(1 - (stretch_max - 1) / 5, stretch_max, num)
            stress_ux, stretch_2_ux, stretch_3_ux = np.array(
                [self._uniaxial(s11) for s11 in stretch_ux]
            ).T
            ux_data = Data(
                "uniaxial", stretch_ux, stretch_2_ux, stretch_3_ux, stress_ux
            )
            out.append(ux_data)

        if bx:
            stretch_bx = np.linspace(1, (stretch_max - 1) / 2 + 1, num)
            stress_bx, stretch_2_bx, stretch_3_bx = np.array(
                [self._biaxial(s11) for s11 in stretch_bx]
            ).T
            bx_data = Data("biaxial", stretch_bx, stretch_2_bx, stretch_3_bx, stress_bx)
            out.append(bx_data)

        if ps:
            stretch_ps = np.linspace(1, stretch_max, num)
            stress_ps, stretch_2_ps, stretch_3_ps = np.array(
                [self._planar(s11) for s11 in stretch_ps]
            ).T
            ps_data = Data("planar", stretch_ps, stretch_2_ps, stretch_3_ps, stress_ps)
            out.append(ps_data)

        return out

    def plot(self, data):

        fig, ax = plt.subplots()

        lineargs = {
            "uniaxial": {"color": "C0"},
            "biaxial": {"color": "C1"},
            "planar": {"color": "C2"},
        }

        for d in data:
            ax.plot(d.stretch, d.stress, **lineargs[d.label], label=d.label)

        ax.grid()
        ax.set_title(self.title)
        ax.set_xlabel(r"stretch $\lambda_1 \quad \longrightarrow$")
        ax.set_ylabel(r"force per undeformed area $P_{11} \quad \longrightarrow$")
        ax.legend()

        fig.tight_layout()

        return fig, ax
