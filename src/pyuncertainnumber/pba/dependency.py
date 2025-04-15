from numbers import Number
from statsmodels.distributions.copula.api import (
    FrankCopula,
    ClaytonCopula,
    GumbelCopula,
    CopulaDistribution,
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
        return f"copula: {self.family}({self.params})"

    def pdf(self, u):
        return self._copula.pdf(u)

    def cdf(self, u):
        return self._copula.cdf(u)

    def sample(self, n: int):
        return self._copula.rvs(n)

    def display(self, ax=None):
        """show the PDF in the u space"""
        self._copula.plot_pdf(ax=ax)

    def fit(self, data):
        return self._copula.fit_corr_param(data)


def supported_family_check(c):
    """check if copula family is supported"""
    if c not in {"gaussian", "t", "frank", "gumbel", "clayton", "independence"}:
        raise Exception("This copula model is not yet implemented")
