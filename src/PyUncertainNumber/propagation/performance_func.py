import numpy as np

""" This module contains the performance functions  """


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
        deflection = F * beam_length**3 / \
            (3 * E * 10**6 * I)  # deflection in m
        stress = F * beam_length * y / I / 1000  # stress in MPa

    except:
        deflection = np.nan
        stress = np.nan

    return np.array([deflection, stress])


def cb_deflection(x):
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
        deflection = F * beam_length**3 / \
            (3 * E * 10**6 * I)  # deflection in m

    except:
        deflection = np.nan

    return deflection


# --------------------- by Leslie ---------------------#

def cb_deflection(beam_length, I, F, E):
    """compute the deflection in the cantilever beam example

    # TODO add typing for UncertainNumber
    Args:
        beam_length (UncertainNumber): Length of the beam (m)
        I: Second moment of area (mm^4)
        F: Applied force (N)
        E: Young's modulus (MPa)

    Returns:
        float: deflection (m)
               Returns np.nan if calculation error occurs.
    """

    deflection = F * beam_length**3 / \
        (3 * E * 10**6 * I)  # deflection in m
    return deflection


def cb_stress(y, beam_length, I, F):
    """to compute bending stress in the cantilever beam example"""

    try:
        stress = F * beam_length * y / I / 1000  # stress in MPa
    except:
        stress = np.nan

    return stress
