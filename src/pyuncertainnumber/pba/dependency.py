from numbers import Number
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import (
    FrankCopula,
    ClaytonCopula,
    GumbelCopula,
    GaussianCopula,
    StudentTCopula,
    IndependenceCopula,
)


class Dependency:

    # parameterisation init
    def __init__(self, family: str, params: Number):
        self.family = family
        self.params = params
        self._post_init_check()
        self._copula = self.copulas_dict.get(self.family)(params)

    copulas_dict = {
        "gaussian": GaussianCopula,
        "t": StudentTCopula,
        "frank": FrankCopula,
        "gumbel": GumbelCopula,
        "clayton": ClaytonCopula,
        "independence": IndependenceCopula,
    }

    def _post_init_check(self):
        supported_family_check(self.family)

    def __repr__(self):
        return f"copula: {self.family} with parameter {self.params}"

    def pdf(self, u):
        return self._copula.pdf(u)

    def cdf(self, u):
        return self._copula.cdf(u)

    def sample(self, n: int):
        """draws n samples in the U space"""
        return self._copula.rvs(n)

    def display(self, style="3d", ax=None):
        """show the PDF in the u space"""
        if style == "2d_pdf":
            self._copula.plot_pdf(ax=ax)
        elif style == "3d_cdf":
            grid_size = 100
            U, V = np.meshgrid(
                np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size)
            )
            Z = np.array(
                [
                    self._copula.cdf([U[i, j], V[i, j]])
                    for i in range(grid_size)
                    for j in range(grid_size)
                ]
            )
            Z = Z.reshape(grid_size, grid_size)
            pl_3d_copula(U, V, Z)
        else:
            raise ValueError("style must be '2d_pdf' or '3d_cdf'")

    def fit(self, data):
        return self._copula.fit_corr_param(data)


def supported_family_check(c):
    """check if copula family is supported"""
    if c not in {"gaussian", "t", "frank", "gumbel", "clayton", "independence"}:
        raise Exception("This copula model is not yet implemented")


def empirical_copula(data):
    """compute the empirical copula"""
    pass


def pl_3d_copula(U, V, Z):

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(U, V, Z, cmap="viridis", edgecolor="none")
    ax.set_xlabel("u")
    ax.set_ylabel("v")
    ax.set_zlabel("C(u, v)")
    plt.show()
