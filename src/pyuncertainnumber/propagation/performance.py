from __future__ import annotations
from typing import TYPE_CHECKING
from numbers import Number
import numpy as np

""" This module contains the some examplar performance/response functions  """


def func_inspect():
    """inspect the formatting of the user-defined function"""
    pass


if TYPE_CHECKING:
    from ..pba.intervals.number import Interval


# * ------------- some of the function for testing APIs


def foo_args(x0, x1, x2):
    """func signature with individual inputs (arguments)"""
    return x0**3 + x1 + x2


def foo(x):
    """func signature with iterable input (list, tuple, np.ndarray)"""
    return x[0] ** 3 + x[1] + x[2]


def foo_vec(x):
    """func signature with matrix input (2D np.ndarray)"""
    return x[:, 0] ** 3 + x[:, 1] + x[:, 2]


def foo_universal(x):
    """a universal signature that takes iterable and matrix inputs but not individual inputs

    caveat:
        - this function works with a vector Interval object
        - but wrong answer with a 2d matrix Interval object due to unclear broadcasting rules
    """
    from ..pba.intervals.number import Interval

    # foo_iterable signature
    if isinstance(x, list | tuple | Interval):
        return x[0] ** 3 + x[1] + x[2]
    elif isinstance(x, np.ndarray):
        if x.ndim == 1:
            x = x[None, :]
        return x[:, 0] ** 3 + x[:, 1] + x[:, 2]


#! 'func' needs to take 2D inputs for maxmising the potential for array computation
def cb_func(x):
    """Calculates deflection and stress for a cantilever beam.

    Args:
        x (np.array): Array of input parameters:
            x[0]: Distance from the neutral axis to the point of interest (m)
            x[1]: Length of the beam (m)
            x[2]: Second moment of area (mm^4)
            x[3]: Applied force (N)
            x[4]: Young's modulus (MPa)

    Returns:
        np.array([deflection (m), stress (MPa)])
               Returns np.array([np.nan, np.nan]) if calculation error occurs.
    """

    y = x[0]
    beam_length = x[1]
    I = x[2]
    F = x[3]
    E = x[4]
    try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
        deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m
        stress = F * beam_length * y / I / 1000  # stress in MPa

    except:
        deflection = np.nan
        stress = np.nan

    return np.array([deflection, stress])


def cb_deflection(x: list[Number] | np.ndarray | Interval | list[Interval]):
    """Calculates deflection and stress for a cantilever beam.

    Args:
        x (np.array): Array of input parameters:
            x[0]: Length of the beam (m)
            x[1]: Second moment of area (mm^4)
            x[2]: Applied force (N)
            x[3]: Young's modulus (MPa)

    Returns:
        float: deflection (m)
               Returns np.nan if calculation error occurs.
    """

    beam_length = x[0]
    I = x[1]
    F = x[2]
    E = x[3]
    try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
        deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m

    except:
        deflection = np.nan

    return deflection


def cb_deflection_vectorised(x: np.ndarray | Interval):
    """Vectorised version for calculating cantilever beam deflection over multiple inputs.

    args:
        x (Interval or np.ndarray) : 2D inputs of  matrix shape, x.shape = (N, 4)
    """

    #! code below works great for np arrays but not for interval objects
    # x = np.atleast_2d(x)  # Ensures x is at least 2D: (N, 4)

    if x.ndim == 1:
        return cb_deflection(x)

    beam_length = x[:, 0]
    I = x[:, 1]
    F = x[:, 2]
    E = x[:, 3]
    deflection = F * beam_length**3 / (3 * E * 10**6 * I)

    if deflection.shape[0] == 1:
        return deflection[0]  # Return scalar
    return deflection


def cb_deflection_separate(L, I, F, E):
    """compute the deflection in the cantilever beam example

    # TODO add typing for UncertainNumber
    Args:
        L (UncertainNumber): Length of the beam (m)
        I: Second moment of area (mm^4)
        F: Applied force (N)
        E: Young's modulus (MPa)

    Returns:
        float: deflection (m)
               Returns np.nan if calculation error occurs.
    """

    deflection = F * L**3 / (3 * E * 10**6 * I)  # deflection in m
    return deflection


def cb_stress(y, L, I, F):
    """to compute bending stress in the cantilever beam example"""

    try:
        stress = F * L * y / I / 1000  # stress in MPa
    except:
        stress = np.nan

    return stress
