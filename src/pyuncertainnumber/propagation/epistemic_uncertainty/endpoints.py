from __future__ import annotations
from typing import TYPE_CHECKING
from ...pba.intervals.number import Interval

import numpy as np
from rich.progress import track
from typing import Callable
from .cartesian_product import cartesian
from ..utils import Propagation_results


# TODO it currently takes 2D numpy array as input
def endpoints_method(
    x: Interval | np.ndarray,
    f: Callable,
    save_raw_data=False,
    results: Propagation_results = None,
) -> Propagation_results:
    """
        Performs uncertainty propagation using the endpoints or vertex method.

    args:
        x (np.ndarray): A 2D NumPy array where each row represents an input variable and
          the two columns define its lower and upper bounds (interval).
        f (Callable): A callable function that takes a 1D NumPy array of input values and
          returns the corresponding output(s).
        results (Propagation_results): The class to use for storing results (defaults to Propagation_results).
        save_raw_data (boolean, optional): Acts as a switch to enable or disable the storage of raw input data when a function (f)
          is not provided.
          - 'False': Returns an error that no function is provided.
          - 'True': Returns the full arrays of unique input combinations.

    notes:
        The function assumes that the intervals in `x` represent uncertainties and aims to provide conservative bounds on the output
        uncertainty.
        If the `f` function returns multiple outputs, the `bounds` array will be 2-dimensional.

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

    raises:
        ValueError if no function is provided and save_raw_data is 'no'.

    example:
        # Example usage with different parameters for minimization and maximization
        f = lambda x: x[0] + x[1] + x[2]  # Example function

        # Determine input parameters for function and method
        x_bounds = np.array([[1, 2], [3, 4], [5, 6]])
        y = endpoints_method(x_bounds, f)

    """
    if isinstance(x, Interval):
        x = x.to_numpy()

    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # Create a sequence of values for each interval based on the number of divisions provided
    # The divisions may be the same for all intervals or they can vary.
    print(
        f"Total number of input combinations for the endpoint method: {2**x.shape[0]}"
    )

    # create an array with the unique combinations of all intervals
    X = cartesian(*x)

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

        if all_output.shape[1] == 1:  # Single output
            results.raw_data["bounds"] = np.array(
                [np.min(all_output, axis=0)[0], np.max(all_output, axis=0)[0]]
            )
        else:  # Multiple outputs
            bounds = np.empty((all_output.shape[1], 2))
            for i in range(all_output.shape[1]):
                bounds[i, :] = np.array(
                    [np.min(all_output[:, i], axis=0), np.max(all_output[:, i], axis=0)]
                )
            results.raw_data["bounds"] = bounds

        for i in range(num_outputs):  # Iterate over outputs
            min_indices = np.where(
                all_output[:, i] == np.min(all_output[:, i], axis=0)
            )[0]
            max_indices = np.where(
                all_output[:, i] == np.max(all_output[:, i], axis=0)
            )[0]

            # Convert to 2D arrays and append
            results.raw_data["min"] = np.append(
                results.raw_data["min"],
                {"x": X[min_indices], "f": np.min(all_output[:, i], axis=0)},
            )
            results.raw_data["max"] = np.append(
                results.raw_data["max"],
                {"x": X[max_indices], "f": np.max(all_output[:, i], axis=0)},
            )

        results.raw_data["f"] = all_output
        results.raw_data["x"] = X

    elif save_raw_data:  # If f is None and save_raw_data is 'yes'
        # Store X in raw_data['x'] even if f is None
        results.add_raw_data(x=X)

    else:
        raise Exception(
            "No function is provided. Select save_raw_data = 'yes' to save the input combinations"
        )

    return results
