import numpy as np


def cartesian(*arrays):
    """Computes the Cartesian product of multiple input arrays

    args:
       - *arrays: Variable number of np.arrays representing the sets of values for each dimension.


    signature:
       - cartesian(*x:np.array)  -> np.ndarray

    note:
       - The data type of the output array is determined based on the input arrays to ensure compatibility.

    return:
        - darray: A NumPy array where each row represents one combination from the Cartesian product.
                  The number of columns equals the number of input arrays.

    example:
        >>> x = np.array([1, 2], [3, 4], [5, 6])
        >>> y = cartesian(*x)
        >>> # Output:
        >>> # array([[1, 3, 5],
        >>> #        [1, 3, 6],
        >>> #        [1, 4, 5],
        >>> #        [1, 4, 6],
        >>> #        [2, 3, 5],
        >>> #        [2, 3, 6],
        >>> #        [2, 4, 5],
        >>> #        [2, 4, 6]])

    """
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, la)
