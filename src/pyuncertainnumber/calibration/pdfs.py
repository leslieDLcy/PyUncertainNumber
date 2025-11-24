"""
@author: Mukesh K. Ramancha

A collection of common probability distributions
"""

import numpy as np
import scipy.stats as stats
from abc import ABC, abstractmethod


class ProbabilityDensityFun(ABC):
    """
    Blueprint for other classes.
    Base class.
    Abstract class is not a concrete class, it cannot be instantiated
    """

    @abstractmethod
    def generate_rns(self, N):
        """
        Method to generate 'N' random numbers

        Parameters
        ----------
        N : int
            number of random numbers needed.

        Returns
        -------
        numpy array of size N

        """
        pass

    @abstractmethod
    def log_pdf_eval(self, x):
        """
        Method to compute log of the pdf at x

        Parameters
        ----------
        x : float
            value where to evalute the pdf.

        Returns
        -------
        float - log of pdf evaluated at x.

        """
        pass


class Uniform(ProbabilityDensityFun):
    """Uniform continuous distribution"""

    def __init__(self, lower=0, upper=1):
        """
        Parameters
        ----------
        lower : float
            lower bound. The default is 0.
        upper : float
            upper bound. The default is 1.
        """
        self.lower = lower
        self.upper = upper

    def generate_rns(self, N):
        return (self.upper - self.lower) * np.random.rand(N) + self.lower

    def log_pdf_eval(self, x):
        if (x - self.upper) * (x - self.lower) <= 0:
            lp = np.log(1 / (self.upper - self.lower))
        else:
            lp = -np.Inf
        return lp


class HalfNormal(ProbabilityDensityFun):
    """Half Normal distribution with zero mean"""

    def __init__(self, sig=1):
        """
        Parameters
        ----------
        sig : float
            standard deviation. The default is 1.
        """
        self.sig = sig

    def generate_rns(self, N):
        return self.sig * np.abs(np.random.randn(N))

    def log_pdf_eval(self, x):
        if x >= 0:
            lp = (
                -np.log(self.sig)
                + 0.5 * np.log(2 / np.pi)
                - ((x * x) / (2 * self.sig * self.sig))
            )
        else:
            lp = -np.Inf
        return lp


class Normal(ProbabilityDensityFun):
    """Normal distribution"""

    def __init__(self, mu=0, sig=1):
        """
        Parameters
        ----------
        mu : float
            mean value. The default is 0.
        sig : float
            standard deviation. The default is 1.
        """
        self.mu = mu
        self.sig = sig

    def generate_rns(self, N):
        return self.sig * np.random.randn(N) + self.mu

    def log_pdf_eval(self, x):
        lp = (
            -0.5 * np.log(2 * np.pi)
            - np.log(self.sig)
            - 0.5 * (((x - self.mu) / self.sig) ** 2)
        )
        return lp


class TruncatedNormal(ProbabilityDensityFun):
    """Truncated Normal distribution"""

    def __init__(self, mu=0, sig=1, low=-np.Inf, up=np.Inf):
        """
        Parameters
        ----------
        mu : float
            mean value. The default is 0.
        sig : float
            standard deviation. The default is 1.
        low : float
            lower bound truncation. The default is -np.Inf.
        up : float
            upper bound truncation. The default is np.Inf.

        """
        self.mu = mu
        self.sig = sig
        self.low = low
        self.up = up

    def generate_rns(self, N):
        return stats.truncnorm(
            (self.low - self.mu) / self.sig,
            (self.up - self.mu) / self.sig,
            loc=self.mu,
            scale=self.sig,
        ).rvs(N)

    def log_pdf_eval(self, x):
        lp = stats.truncnorm(
            (self.low - self.mu) / self.sig,
            (self.up - self.mu) / self.sig,
            loc=self.mu,
            scale=self.sig,
        ).logpdf(x)
        return lp


class MultivariateNormal(ProbabilityDensityFun):
    """Multivariate Normal distribution"""

    def __init__(self, mu=np.zeros(2), E=np.identity(2)):
        """
        Parameters
        ----------
        mu : np array
            mean vector. The default is np.zeros(2).
        E : np 2D array
            covariance matrix. The default is np.identity(2).
        """
        self.mu = mu
        self.E = E
        self.d = len(mu)
        self.logdetE = np.log(np.linalg.det(self.E))
        self.Einv = np.linalg.inv(E)

    def generate_rns(self, N):
        return np.random.multivariate_normal(self.mu, self.E, N)

    def log_pdf_eval(self, x):
        xc = x - self.mu
        return (
            -(0.5 * self.d * np.log(2 * np.pi))
            - (0.5 * self.logdetE)
            - (0.5 * np.transpose(xc) @ self.Einv @ xc)
        )
