import numpy as np


class BazantOh:
    def __init__(self, n: int = 21):
        """ "Points and weights of a numeric integration scheme on the surface
        of a sphere.

        Bazant, Z. P., & Oh, B. H. (1986). Efficient Numerical Integration on
        the Surface of a Sphere. ZAMM ‐ Journal of Applied Mathematics and
        Mechanics / Zeitschrift für Angewandte Mathematik und Mechanik, 66(1),
        37-49. https://doi.org/10.1002/zamm.19860660108

        """

        schemes = {
            21: self._scheme_21,
        }

        self.points, self.weights = schemes[n]()

    def _scheme_21(self):
        "2x21-point scheme (degree 9, orthogonal symmetries)."

        a = np.sqrt(2) / 2
        b = 0.836095596749
        c = 0.387907304067

        points = np.array(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, a, a],
                [0, -a, a],
                [a, 0, a],
                [-a, 0, a],
                [a, a, 0],
                [-a, a, 0],
                [b, c, c],
                [-b, c, c],
                [b, -c, c],
                [-b, -c, c],
                [c, b, c],
                [-c, b, c],
                [c, -b, c],
                [-c, -b, c],
                [c, c, b],
                [-c, c, b],
                [c, -c, b],
                [-c, -c, b],
            ]
        ).T

        w1 = 0.0265214244093
        w2 = 0.0199301476312
        w3 = 0.0250712367487

        weights = 2 * np.concatenate(
            [np.repeat(w1, 3), np.repeat(w2, 6), np.repeat(w3, 12)]
        )

        return points, weights
