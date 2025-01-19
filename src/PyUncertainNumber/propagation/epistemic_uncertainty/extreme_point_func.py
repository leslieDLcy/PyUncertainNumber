import numpy as np


def extreme_pointX(ranges, signX):
    """ Calculates the extreme points of a set of ranges based on signs.

    args:
        ranges: A NumPy array of shape (d, 2) representing the ranges 
                 (each row is a variable, each column is a bound).
        signX: A NumPy array of shape (1, d) representing the signs.

    returns:
        A NumPy array of shape (2, d) representing the extreme points.
    """
    d = len(ranges)  # Get the number of dimensions
    pts = np.tile(signX, (2, 1))  # Repeat signX twice vertically

    Xsign = np.zeros((2, d))
    for j in range(d):  # Iterate over dimensions
        if pts[0, j] > 0:  # Check sign for the first row (minimum)
            Xsign[0, j] = ranges[j][0]  # Take the first element (lower bound)
            Xsign[1, j] = ranges[j][-1]  # Take the last element (upper bound)
        else:
            Xsign[0, j] = ranges[j][-1]  # Take the last element (upper bound)
            Xsign[1, j] = ranges[j][0]  # Take the first element (lower bound)

    return Xsign
