import numpy as np
import tqdm
from typing import Callable
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import (
    cartesian,
)
from pyuncertainnumber.propagation.utils import Propagation_results


def subinterval_method(
    x: np.ndarray,
    f: Callable,
    results: Propagation_results = None,
    n_sub: np.array = 3,
    save_raw_data="no",
) -> Propagation_results:  # Specify return type
    """subinterval reconstitution method

    args:
        - x (nd.array): A 2D NumPy array where each row represents an input variable and the two columns
            define its lower and upper bounds (interval).
        - f (callable): A callable function that takes a 1D NumPy array of input values and returns the
            corresponding output(s).
        - n_sub (nd.array): A scalar (integer) or a 1D NumPy array specifying the number of subintervals for
            each input variable.
             - If a scalar, all input variables are divided into the same number of subintervals (defaults 3 divisions).
             - If an array, each element specifies the number of subintervals for the
               corresponding input variable.
        - save_raw_data (boolean): Controls the amount of data returned:
             - 'no': Returns only the minimum and maximum output values along with the
                    corresponding input values.
             - 'yes': Returns the above, plus the full arrays of unique input combinations
                      (`all_input`) and their corresponding output values (`all_output`).

    signature:
        subinterval_method(x:np.ndarray, f:Callable, n:np.array, results:dict = None, save_raw_data = 'no') -> dict

    note:
        - The function assumes that the intervals in `x` represent uncertainties and aims to provide conservative
           bounds on the output uncertainty.
        - The computational cost increases exponentially with the number of input variables
           and the number of subintervals per variable.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional.

    return:
         - dict: A dictionary containing the results:
           - 'bounds': An np.ndarray of the bounds for each output parameter (if f is not None).
        - 'min': A dictionary for lower bound results (if f is not None):
             - 'x': Input values that produced the minimum output value(s).
             - 'f': Minimum output value(s).
        - 'max': A dictionary for upper bound results (if f is not None):
             - 'x': Input values that produced the maximum output value(s).
             - 'f': Maximum output value(s).
        - 'raw_data': A dictionary containing raw data (if `save_raw_data` is 'yes'):
             - 'x': All generated input samples.
             - 'f': Corresponding output values for each input sample.

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
    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # Create a sequence of values for each interval based on the number of divisions provided
    # The divisions may be the same for all intervals or they can vary.
    m = x.shape[0]
    print(
        f"Total number of input combinations for the subinterval method: {(n_sub+1)**m}"
    )

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
        all_output = np.array([f(xi) for xi in tqdm.tqdm(X, desc="Evaluating samples")])

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

    elif save_raw_data == "yes":  # If f is None and save_raw_data is 'yes'
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

# #Print the results
# y.print()
