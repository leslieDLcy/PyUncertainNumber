import numpy as np
from typing import Callable
from numpy import ndarray

##### the module #####

def endpoints_propagation_2n(
    x: ndarray, f: Callable
):  
    """ Marco's original implementation of the endpoints propagation method.
    
    note:
        - Computes the min and max of a monotonic function with endpoints propagation
        - "x has shape (n,2)."
    """
    # ! if it only works on monotonic functions, it should be called minmax_monotonic?
    # I'm using it in the `UN.endpointMethod` method.

    n = x.shape[0]
    max_candidate = -np.inf
    min_candidate = np.inf
    for j in range(2**n):
        index = tuple(
            [j // 2**h - (j // 2 ** (h + 1)) * 2 for h in range(n)]
        )  # tuple of 0s and 1s
        itb = index_to_bool_(index).T
        new_f = f(*x[itb])
        if new_f > max_candidate:
            max_candidate = new_f
            max_corner = index
        if new_f < min_candidate:
            min_candidate = new_f
            min_corner = index
    return min_candidate, max_candidate, min_corner, max_corner


def index_to_bool_(index: ndarray, dim=2):
    """Turns a vector of indices e.g. (1,0,0,0,1) to an array of boolean [(0,1),(1,0),(1,0),(1,0),(0,1)] for masking.
    If dim > 2,  e.g. (2,0,1,0) the array of booleans is [(0,0,1),(1,0,0),(0,1,0),(1,0,0)].
    """
    index = np.asarray(index, dtype=int)
    return np.asarray([index == j for j in range(dim)], dtype=bool)


def main():

    def linearfun(*x):
        return sum(x)

    def a(x):
        return np.asarray(x, dtype=float)

    D1 = a(
        [
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
    )

    X = D1
    print(endpoints_propagation_2n(X, linearfun))

    ### test ###

    # x1=a([1,2])
    # x2=a([2,3])
    # x3=a([1,1])
    # x4=a([4,5])

    # xx = i(24*[x1])
    # Xv = X.val

    # print(xx_.shape)
    # print(linearfun(x1,x2,x3,x4))
    # print(compute.max_bruteforce(X.T))

    # X  = D2
    # X = D3_

    # print(X)

    # X = fuzzy_structure_(core=[2.0,3.0],range=[1.0,9.0],steps=400)

    # X = D9

    # X = np.tile(X,(100,1))


if __name__ == "__main__":
    main()
