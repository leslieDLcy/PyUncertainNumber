import numpy as np
from .methods import lo, hi, mid, rad, width, intervalise

""" this file contains the functions to create interval matrices from numpy arrays 

TODO: func `intervalise()` change the default setting to consume by the last axis to first axis or 2by2 matrix will be wrong;
"""


def consume_interval(low, high):
    """initialise an interval matrix from `low, high` arrays;

    note:
        initialise from the shape (2, m, n) as preferred;

    example:
        low = np.arange(9).reshape(3,3)
        high = np.arange(10, 19).reshape(3,3)
    """

    a_matrix = np.stack([low, high], axis=0)
    return intervalise(a_matrix)


def create_interval(matrix, half_width=0.1):
    """mannually create an interval matrix from a numpy array"""

    low = matrix - half_width
    high = matrix + half_width
    return consume_interval(low, high)


def dot(x, y):
    return sum(x * y)


def rowcol(W, x):
    """marco's original implementation of the rowcol function"""

    s = W.shape
    y = []
    for i in range(s[0]):
        y.append(dot(W[i], x))
    return intervalise(y)


def rowcol2(x, W):
    """Leslie changed the argument order for notational convecience original implementation of the rowcol function

    args:
        - W: have to be a vector;
    """

    s = x.shape
    if len(s) > 1:
        y = []
        for i in range(s[0]):
            y.append(dot(x[i], W))

        return intervalise(y)
    else:
        print("x shall be a 2 dimensional array")


# def rowcol_wrong(x, W):
#     """ Leslie's implementation of the rowcol function

#     args:
#         - x: a vector, e.g. hidden layer output, as a row vector
#         - W: weight matrix of the next layer

#     note:
#         - this is not full-fleged interval matrix computation
#         - it currently only fits for `x` as a vector
#         - it can be used for hidden-layer tensor propagation
#     """

#     s = W.shape
#     y=[]
#     for j in range(s[1]):
#         y.append(dot(x, W[:, j]))
#         result = intervalise(y)
#     return result[np.newaxis, :]


def consume_list(list_intervals):
    """consume a list of interval matrices into a single interval matrix

    ! of no use now
    note:

        - being used for interval matrix multiplication
    """

    low, upper = [], []
    for interval in list_intervals:
        low.append(interval.lo)
        upper.append(interval.hi)

    low_a = np.vstack(low)
    upper_a = np.vstack(upper)

    return intervalise(low_a, upper_a)


def intvl_matmul(x, W):
    """draft matrix multiplication function for interval matrices

    note:
        - can be used for general matrix multiplication

    return:
        - an interval matrix
    """

    row_cp_list = []
    sW = W.shape
    if (len(sW)) > 1:
        if sW[1] > 1:
            for j in range(sW[1]):
                row_cp_list.append(rowcol2(x, W[:, j]))
            return intervalise(row_cp_list)
        elif sW[1] == 1:
            return rowcol2(x, W[:, 0])
        else:
            return rowcol2(x, W)
    else:
        return rowcol2(x, W)
