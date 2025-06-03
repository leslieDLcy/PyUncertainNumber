import numpy as np
import tqdm
from typing import Callable
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import (
    cartesian,
)
from pyuncertainnumber.propagation.epistemic_uncertainty.extreme_point_func import (
    extreme_pointX,
)
from pyuncertainnumber.propagation.utils import Propagation_results


def extremepoints_method(
    x: np.ndarray, f: Callable, results: Propagation_results = None, save_raw_data=False
) -> Propagation_results:  # Specify return type
    """
        Performs uncertainty propagation using the Extreme Point Method for monotonic functions.
        This method estimates the bounds of a function's output by evaluating it at specific combinations of extreme values
        (lower or upper bounds) of the input variables. It is efficient for monotonic functions but might not be
        accurate for non-monotonic functions.

    args:
        x (np.ndarray): A 2D NumPy array where each row represents an input variable and
          the two columns define its lower and upper bounds (interval).
        f (Callable): A callable function that takes a 1D NumPy array of input values and
          returns the corresponding output(s).
        results (Propagation_results): The class to use for storing results (defaults to Propagation_results).
        save_raw_data (str, optional): Acts as a switch to enable or disable the storage of raw input data
            when a function (f)
            is not provided.
            - 'no': Returns an error that no function is provided.
            - 'yes': Returns the full arrays of unique input combinations.

    signature:
        extremepoints_method(x:np.ndarray, f:Callable, results:dict, save_raw_data = 'no') -> dict

    raises:
        ValueError if no function is provided and save_raw_data is 'no'.

    return:
        Returns `Propagation_results` object(s) containing:
            - 'un': UncertainNumber object(s) to characterise the interval(s) of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output values.
                    - 'min' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the minimum of that output.
                    - 'max' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the maximum of that output.
                    - 'bounds' (np.ndarray): 2D array of lower and upper bounds for each output.

    Example:
        # Example usage with different parameters for minimization and maximization
        >>> f = lambda x: x[0] + x[1] + x[2]  # Example function
        >>> # Determine input parameters for function and method
        >>> x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
        >>> # Call the method
        >>> y = extremepoint_method(x_bounds, f)
        >>> # print results
        >>> y.print()

    """
    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # create an array with the unique combinations of all intervals
    X = cartesian(*x)

    d = X.shape[1]  # Number of dimensions
    inds = np.array([1] + [2**i + 1 for i in range(d)])  # Generate indices
    # Select rows based on indices (adjusting for 0-based indexing)
    Xeval = X[inds - 1]

    # propagates the epistemic uncertainty through extremepoints
    if f is not None:

        # Simulate function for the selected subset
        all_output = []
        for c in tqdm.tqdm(Xeval, desc="Number of function evaluations"):
            output = f(c)
            all_output.append(output)

        # Determine the number of outputs from the first evaluation
        try:
            num_outputs = len(all_output[0])
        except TypeError:
            num_outputs = 1  # If f returns a single value

        # Convert all_output to a NumPy array with the correct shape
        all_output = np.array(all_output).reshape(-1, num_outputs)

        # Calculate signs
        part_deriv_sign = np.zeros((num_outputs, d))
        Xsign = np.zeros((2 * num_outputs, d))
        for i in range(num_outputs):
            # Calculate signs based on initial output values
            part_deriv_sign[i] = np.sign(all_output[1:, i] - all_output[0, i])[::-1]

            # Calculate extreme points
            Xsign[2 * i : 2 * (i + 1), :] = extreme_pointX(x, part_deriv_sign[i])

        # Store and reuse evaluations
        evaluated_points = {}
        lower_bound = np.full(num_outputs, np.inf)  # Initialize with inf
        upper_bound = np.full(num_outputs, -np.inf)  # Initialize with -inf
        min_indices = np.zeros((d, num_outputs))
        max_indices = np.zeros((d, num_outputs))
        num_evaluations = 0  # Initialize evaluation counter

        for i in range(num_outputs):
            for j in range(2):  # For lower and upper bounds
                key = tuple(Xsign[2 * i + j, :])
                if key not in evaluated_points:
                    evaluated_points[key] = f(Xsign[2 * i + j, :])
                    num_evaluations += 1
                if num_outputs == 1:
                    output = evaluated_points[key]

                else:
                    output = evaluated_points[key][i]

                # Determine if this is the lower or upper bound for this output
                if output < lower_bound[i]:
                    lower_bound[i] = output
                    min_indices[:, i] = Xsign[2 * i + j, :]
                if output > upper_bound[i]:
                    upper_bound[i] = output
                    max_indices[:, i] = Xsign[2 * i + j, :]

        print(f"Number of total function evaluations: {num_evaluations + len(Xeval)}")
        # Convert to 2D arrays (if necessary) and append
        for i in range(num_outputs):
            results.raw_data["min"] = np.append(
                results.raw_data["min"], {"x": min_indices[:, i], "f": lower_bound[i]}
            )
            results.raw_data["max"] = np.append(
                results.raw_data["max"], {"x": max_indices[:, i], "f": upper_bound[i]}
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

        results.add_raw_data(part_deriv_sign=part_deriv_sign)
        results.raw_data["x"] = Xeval
        results.raw_data["f"] = all_output

    elif save_raw_data:  # If f is None and save_raw_data is 'yes'
        # Store Xeval in raw_data['x'] even if f is None
        results.add_raw_data(x=Xeval)

    else:
        raise ValueError(
            "No function is provided. Select save_raw_data = 'yes' to save the input combinations"
        )

    return results  # only the interval
