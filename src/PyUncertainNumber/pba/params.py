""" Hyperparameters for the pba """

from dataclasses import dataclass
import numpy as np
import scipy.stats as sps
""" hyperparameters for the pba """


@dataclass
class ApproximatorRegCoefficients:
    A: float
    B: float
    C: float
    D: float
    E: float
    F: float
    G: float
    H: float
    sigma: float

    @staticmethod
    def lognormal(m, s):
        m2 = m**2
        s2 = s**2
        mlog = np.log(m2/np.sqrt(m2+s2))
        slog = np.sqrt(np.log((m2+s2)/m2))
        return (sps.lognorm.rvs(s=slog, scale=np.exp(mlog), size=2000))

    @staticmethod
    def env(x, y): return np.concatenate((x, y))

    def _cp(self, z, r, f):
        self.L = self.A + self.B * z + self.C * r + self.D * f + self.E * \
            z * r + self.F * z * f + self.G * r * f + self.H * z * r * f
        self.w = 10**self.L
        self.a = 10**z + self.w / 2 * np.array([-1, 1])
        self.q = self.lognormal(m=10**(self.sigma**2/2),
                                s=np.sqrt(10**(2*self.sigma**2) -
                                          10**(self.sigma**2)),
                                )
        # self.p = self.env(min(self.a) - self.q, self.q + max(self.a))
        self.p = (min(self.a) - self.q, self.q + max(self.a))


@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:

    about = ApproximatorRegCoefficients(-0.2085, 0.4285, 0.2807,
                                        0.0940, 0.0147, -0.0640, -0.0102, 0.0404, 0.5837)

    steps = 200
    many = 2000
    # the percentiles
    p_values = np.linspace(0.0001, 0.9999, steps)

    p_lboundary = 0.0001
    p_hboundary = 0.9999

    # by default
    scott_hedged_interpretation = {}

    # user-defined
    user_hedged_interpretation = {}

    result_path = "./results/"
    hw = 0.5  # default half-width during an interval instantiation via PM method
    # @property
    # # template for property
    # def sth(self):
    #     """ template for property"""
    #     return int(round(self.patch_window_seconds / self.stft_hop_seconds))


@dataclass(frozen=True)  # Instances of this class are immutable.
class Data:

    # scott construct p28
    skinny = [
        [1.0, 1.52],
        [2.68, 2.98],
        [7.52, 7.67],
        [7.73, 8.35],
        [9.44, 9.99],
        [3.66, 4.58]
    ]

    puffy = [
        [3.5, 6.4],
        [6.9, 8.8],
        [6.1, 8.4],
        [2.8, 6.7],
        [3.5, 9.7],
        [6.5, 9.9],
        [0.15, 3.8],
        [4.5, 4.9],
        [7.1, 7.9]
    ]

    sudret = [4.02, 4.07, 4.25, 4.32, 4.36, 4.45, 4.47,
              4.57, 4.58, 4.62, 4.68, 4.71, 4.72, 4.79,
              4.85, 4.86, 4.88, 4.90, 5.08, 5.09, 5.29,
              5.30, 5.40, 5.44, 5.59, 5.59, 5.70, 5.89,
              5.89, 6.01]

    # from Scott Ioanna5.py
    k = 22
    m = 11
    n = k + m
    fdata = np.concatenate((m*[0], k*[1]))
    bdata = np.random.uniform(size=25) > 0.35
    idata = np.round(np.random.uniform(size=25) * 16)
    data = np.random.uniform(size=25) * 30
    x2 = 5 + np.random.uniform(size=25) * 30
    error = np.random.normal(size=25)

    # @property
    # # template for property
    # def sth(self):
    #     """ template for property"""
    #     return int(round(self.patch_window_seconds / self.stft_hop_seconds))


@dataclass(frozen=True)  # Instances of this class are immutable.
class Named:

    k = 22
    m = 11
    n = k + m

    # @property
    # # template for property
    # def sth(self):
    #     """ template for property"""
    #     return int(round(self.patch_window_seconds / self.stft_hop_seconds))
