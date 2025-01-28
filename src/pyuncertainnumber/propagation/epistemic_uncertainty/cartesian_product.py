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


# def cartesian(sides): to be deleted perhaps
#   """
#   Generates the Cartesian product of the input sets.

#   Args:
#     sides: A NumPy array where each row represents a set of values.

#   Returns:
#     A NumPy array containing the Cartesian product of the input sets,
#     where each row represents a unique combination.
#   """
#   n = sides.shape[0]  # Number of sets
#   F = np.meshgrid(*sides)  # Generate the ndgrid
#   G = np.zeros((np.prod([len(s) for s in sides]), n))  # Initialize output array
#   for i in range(n):
#     G[:, i] = F[i].flatten()  # Flatten and assign to output array

#   C = np.unique(G, axis=0)  # Find unique rows
#   return C

# # # example
# # # Define the sets
# # sides = np.array([
# #     [1, 2],    # First set
# #     [10, 20],  # Second set
# #     [100, 200] # Third set
# # ])

# # # Generate the Cartesian product
# # C = cartesian(sides)

# # # Print the result
# # print(C)
