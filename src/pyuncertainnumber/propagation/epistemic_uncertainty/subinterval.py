from __future__ import annotations
from typing import TYPE_CHECKING
from ...pba.intervals.number import Interval

import numpy as np
from rich.progress import track
from typing import Callable
from .cartesian_product import cartesian
from ..utils import Propagation_results


def subinterval_method(
    x: np.ndarray,
    f: Callable,
    save_raw_data=False,
    n_sub: np.array = 10,
    results: Propagation_results = None,
) -> Propagation_results:  # Specify return type
    """The subinterval reconstitution method.

    args:
        x (nd.array): A 2D NumPy array where each row represents an input variable and the two columns
            define its lower and upper bounds (interval).
        f (callable): A callable function that takes a 1D NumPy array of input values and returns the
            corresponding output(s).
        results (Propagation_results): The class to use for storing results (defaults to Propagation_results).
        n_sub (nd.array): A scalar (integer) or a 1D NumPy array specifying the number of subintervals for
            each input variable.
            - If a scalar, all input variables are divided into the same number of subintervals (defaults 3 divisions).
            - If an array, each element specifies the number of subintervals for the
               corresponding input variable.
        save_raw_data (str, optional): Acts as a switch to enable or disable the storage of raw input data when a function (f)
            is not provided.
            - 'no': Returns an error that no function is provided.
            - 'yes': Returns the full arrays of unique input combinations.

    signature:
        subinterval_method(x:np.ndarray, f:Callable, n_sub:np.array ...) -> Propagation_results

    notes:
        - The function assumes that the intervals in `x` represent epistemic uncertainties in the input.
        - The subinterval reconstitution method subdivides the input intervals into smaller subintervals
          to accommodate for the presence of non-monotonic trends in the function output(s).
        - The subintervals for the input can vary in number.
        - The computational cost increases exponentially with the number of input variables
          and the number of subintervals per variable.
        - The `f` function can return multiple outputs.

    raises:
        ValueError if no function is provided and save_raw_data is 'no'.

    returns:
        `Propagation_results` object(s) containing:
            - 'un': UncertainNumber object(s) to characterise the interval(s) of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output values.
                    - 'min' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the minimum of that output.
                    - 'max' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the maximum of that output.
                    - 'bounds' (np.ndarray): 2D array of lower and upper bounds for each output.

    example:
        >>> #Define input intervals
        >>> x = np.array([[1, 2], [3, 4], [5, 6]])
        >>> # Define the function
        >>> f = lambda x: x[0] + x[1] + x[2]
        >>> # Run sampling method with n = 2
        >>> y = subinterval_method(x, f, n_sub, save_raw_data = 'yes')
        >>> # Print the results
        >>> y.print()
    """
    if isinstance(x, Interval):
        x = x.to_numpy()

    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # Create a sequence of values for each interval based on the number of divisions provided
    # The divisions may be the same for all intervals or they can vary.
    m = x.shape[0]

    if type(n_sub) == int:  # All inputs have identical division
        total = (n_sub + 1) ** m
        Xint = np.zeros((0, n_sub + 1), dtype=object)
        for xi in x:
            new_X = np.linspace(xi[0], xi[1], n_sub + 1)
            Xint = np.concatenate((Xint, new_X.reshape(1, n_sub + 1)), axis=0)
    else:  # Different divisions per input
        total = 1
        Xint = []
        for xc, c in zip(x, range(len(n_sub))):
            total *= n_sub[c] + 1
            new_X = np.linspace(xc[0], xc[1], n_sub[c] + 1)

            Xint.append(new_X)

        Xint = np.array(Xint, dtype=object)
    # create an array with the unique combinations of all subintervals
    X = cartesian(*Xint)

    # propagates the epistemic uncertainty through subinterval reconstitution
    if f is not None:
        all_output = np.array(
            [f(xi) for xi in track(X, description="Function evaluations")]
        )

        try:
            num_outputs = len(all_output[0])
        except TypeError:
            num_outputs = 1  # If f returns a single value

        # Reshape all_output to a 2D array (Corrected)
        all_output = np.array(all_output).reshape(-1, num_outputs)

        # Calculate lower and upper bounds for each output
        lower_bound = np.min(all_output, axis=0)
        upper_bound = np.max(all_output, axis=0)

        # Find indices of minimum and maximum values for each output
        min_indices = np.array(
            [
                np.where(all_output[:, i] == lower_bound[i])[0]
                for i in range(num_outputs)
            ]
        )
        max_indices = np.array(
            [
                np.where(all_output[:, i] == upper_bound[i])[0]
                for i in range(num_outputs)
            ]
        )

        # Convert to 2D arrays (if necessary) and append
        for i in range(num_outputs):
            results.raw_data["min"] = np.append(
                results.raw_data["min"], {"x": X[min_indices[i]], "f": lower_bound[i]}
            )
            results.raw_data["max"] = np.append(
                results.raw_data["max"], {"x": X[max_indices[i]], "f": upper_bound[i]}
            )
            results.raw_data["bounds"] = (
                np.vstack(
                    [
                        results.raw_data["bounds"],
                        np.array([lower_bound[i], upper_bound[i]]),
                    ]
                )
                if results.raw_data["bounds"].size
                else np.array([lower_bound[i], upper_bound[i]])
            )

        results.raw_data["f"] = all_output
        results.raw_data["x"] = X

    elif save_raw_data:  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(x=X)
    else:
        print(
            "No function is provided. Select save_raw_data = 'yes' to save the input combinations"
        )

    return results


# # Example usage with different parameters for minimization and maximization
# f = lambda x: x[0] + x[1] + x[2]  # Example function

# # Determine input parameters for function and method
# x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
# n=2
# # Call the method
# y = subinterval_method(x_bounds, f=None, n_sub=n, save_raw_data = 'yes')
