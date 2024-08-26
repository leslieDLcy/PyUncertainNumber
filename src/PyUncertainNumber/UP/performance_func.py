import numpy as np

""" This module contains the performance functions  """


def cantilever_beam_func(x):
    """the function of the cantilever beam example"""

    y = x[0]
    beam_length = x[1]
    I = x[2]
    F = x[3]
    E = x[4]
    try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
        deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m
        stress = F * beam_length * y / I / 1000  # stress in MPa
        # print(f'Successully completed eval #{i} of total {total}...')
    except:
        # print(f'Error in eval #{i} of total {total}...')
        deflection = np.nan
        stress = np.nan

    return [deflection, stress]


# return dictTuple


def cantilever_beam_deflection(beam_length, I, F, E):
    """to compute the deflection in the cantilever beam example"""

    try:
        deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m
    except:
        deflection = np.nan

    return deflection


def cantilever_beam_stress(y, beam_length, I, F):
    """to compute bending stress in the cantilever beam example"""

    try:
        stress = F * beam_length * y / I / 1000  # stress in MPa
    except:
        stress = np.nan

    return stress
