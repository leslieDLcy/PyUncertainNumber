""" Hyperparameters for the pba """

from dataclasses import dataclass
import numpy as np

""" hyperparameters for the pba """


@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:

    hedge_cofficients = {
        "about": [
            -0.2085,
            0.4285,
            0.2807,
            0.0940,
            0.0147,
            -0.0640,
            -0.0102,
            0.0404,
            0.5837,
        ],
        "roughly": [
            -0.103,
            0.3687,
            0.2559,
            -0.0303,
            0.0353,
            0.1051,
            0.1422,
            -0.0562,
            0.5966,
        ],
        "approximately": [
            -0.3171,
            0.4993,
            0.254,
            0.6410,
            0.0177,
            -0.2835,
            0.1025,
            0.0169,
            0.6192,
        ],
        "around": [
            -0.1018,
            0.3429,
            0.3169,
            0.0951,
            0.0381,
            -0.0005,
            0.0174,
            -0.0029,
            0.5261,
        ],
        "at most": [
            -0.3076,
            0.4751,
            0.2477,
            0.1168,
            0.0088,
            -0.0619,
            0.1551,
            0.0052,
            0.5956,
        ],
        "at least": [
            -0.1128,
            0.3624,
            0.3829,
            0.3188,
            0.0087,
            -0.1404,
            0.0069,
            0.0409,
            0.5927,
        ],
        "no more than": [
            -0.2699,
            0.4187,
            0.2418,
            0.2216,
            0.0382,
            -0.0467,
            0.0689,
            0.0069,
            0.5916,
        ],
        "no less than": [
            -0.0187,
            0.3412,
            0.2207,
            -0.1427,
            0.0341,
            0.1616,
            -0.1480,
            0.0083,
            0.6314,
        ],
        "over": [
            -0.0668,
            0.3793,
            0.2490,
            -0.1635,
            0.0344,
            0.1353,
            -0.0068,
            -0.0176,
            0.6666,
        ],
        "above": [
            -0.1736,
            0.4483,
            0.2625,
            0.0224,
            0.0112,
            0.0669,
            -0.1354,
            0.0156,
            0.6668,
        ],
        "below": [
            -0.3052,
            0.4275,
            0.2666,
            0.3141,
            0.0353,
            -0.1348,
            0.1678,
            -0.0168,
            0.6577,
        ],
        "almost": [
            -0.4539,
            0.4593,
            0.3567,
            0.4006,
            -0.0196,
            -0.2245,
            -0.0534,
            0.0882,
            0.6640,
        ],
        "nearly": [
            -0.2716,
            0.3420,
            0.2722,
            0.0923,
            0.0440,
            0.0196,
            0.0386,
            -0.0270,
            0.5969,
        ],
        "": [0.2070, 0.1374, -0.4265, -0.4267, 0.2450, 0.3341, 2.1650, -0.6876, 0.7869],
        "precisely": [
            -0.4989,
            0.5884,
            0.3812,
            1.3500,
            -0.0774,
            -0.8274,
            -0.7248,
            0.2464,
            0.3859,
        ],
        "exactly": [
            -0.8360,
            0.7434,
            0.6058,
            5.427,
            -0.2055,
            -0.7757,
            0.0000,
            0.0000,
            1.0370,
        ],
    }

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
        [3.66, 4.58],
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
        [7.1, 7.9],
    ]

    sudret = [
        4.02,
        4.07,
        4.25,
        4.32,
        4.36,
        4.45,
        4.47,
        4.57,
        4.58,
        4.62,
        4.68,
        4.71,
        4.72,
        4.79,
        4.85,
        4.86,
        4.88,
        4.90,
        5.08,
        5.09,
        5.29,
        5.30,
        5.40,
        5.44,
        5.59,
        5.59,
        5.70,
        5.89,
        5.89,
        6.01,
    ]

    # from Scott Ioanna5.py
    k = 22
    m = 11
    n = k + m
    fdata = np.concatenate((m * [0], k * [1]))
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
