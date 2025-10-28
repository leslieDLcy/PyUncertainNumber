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


from numbers import Number
from statsmodels.distributions.copula.api import (
    GaussianCopula,
    StudentTCopula,
    FrankCopula,
    GumbelCopula,
    ClaytonCopula,
    IndependenceCopula,
)
import inspect


class Dependency:
    """
    Dependency class to specify copula models.

    Args:
        family (str): Name of the copula family, one of
            "gaussian", "t", "frank", "gumbel", "clayton", "independence".
        params (Number | None): Backward-compatible single-parameter shortcut:
            - gaussian/t: interpreted as corr
            - frank/gumbel/clayton: interpreted as theta
            - independence: ignored
        **kwargs: Any keyword parameters supported by the selected copula,
            e.g. corr=..., df=..., theta=..., k_dim=..., allow_singular=...

    Examples
    --------
    >>> Dependency("gaussian", params=0.8, k_dim=3)          # legacy style
    >>> Dependency("gaussian", corr=0.8, k_dim=3)            # explicit
    >>> Dependency("t", corr=0.6, df=5, k_dim=4)
    >>> Dependency("frank", theta=2.5, k_dim=2)
    >>> Dependency("independence", k_dim=5)
    """

    copulas_dict = {
        "gaussian": GaussianCopula,
        "t": StudentTCopula,
        "frank": FrankCopula,
        "gumbel": GumbelCopula,
        "clayton": ClaytonCopula,
        "independence": IndependenceCopula,
    }

    # legacy single-parameter aliasing
    _single_param_alias = {
        "gaussian": "corr",
        "t": "corr",
        "frank": "theta",
        "gumbel": "theta",
        "clayton": "theta",
        "independence": None,
    }

    def __init__(self, family: str, params: Number | None = None, **kwargs):
        fam = str(family).strip().lower()
        if fam not in self.copulas_dict:
            raise ValueError(
                f"Unknown copula family '{family}'. "
                f"Choose from {list(self.copulas_dict)}."
            )

        # Map legacy `params` to the family-specific keyword if user didn't supply it.
        alias = self._single_param_alias[fam]
        if params is not None and alias is not None and alias not in kwargs:
            kwargs[alias] = params

        Copula = self.copulas_dict[fam]

        # Keep only kwargs that the Copula actually accepts (and error on unknowns)
        sig = inspect.signature(Copula.__init__)
        valid_keys = set(sig.parameters.keys()) - {"self"}
        unknown = set(kwargs.keys()) - valid_keys
        if unknown:
            raise TypeError(
                f"Got unexpected arguments for {Copula.__name__}: {sorted(unknown)}. "
                f"Valid kwargs are: {sorted(valid_keys)}."
            )

        # Instantiate and store
        self.family = fam
        self.params = params
        self._copula = Copula(**{k: kwargs[k] for k in kwargs if k in valid_keys})

    @property
    def copula(self):
        """Access the underlying statsmodels copula instance."""
        return self._copula

    def _post_init_check(self):
        supported_family_check(self.family)

    def __repr__(self):
        return f"copula: {self.family} with parameter {self.params}"

    def pdf(self, u):
        return self._copula.pdf(u)

    def cdf(self, u):
        return self._copula.cdf(u)

    def u_sample(self, n: int, random_state=None):
        """draws n samples in the U space (unit hypercube)"""
        rd = 42 if random_state is None else random_state
        return self._copula.rvs(n, random_state=rd)

    def display(self, style="3d_cdf", ax=None):
        """show the PDF or CDF in the u space"""
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
