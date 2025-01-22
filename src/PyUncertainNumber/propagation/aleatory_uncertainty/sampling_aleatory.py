import numpy as np
import tqdm
from typing import Callable
from scipy.stats import qmc  # Import Latin Hypercube Sampling from SciPy
from PyUncertainNumber.propagation.utils import Propagation_results

def sampling_aleatory_method(
    x: list,
    f: Callable,
    results: Propagation_results = None,
    n_sam: int = 500,
    method="monte_carlo",
    save_raw_data="no") -> Propagation_results:  # Specify return type
    """Performs uncertainty propagation using Monte Carlo or Latin Hypercube sampling, similar to the `sampling_method`.
       It also combines the sampling method with the ednpoints method.  

    description:
        - Propagates aleatory uncertainty in the input by generating random samples from the input 
        distributions and evaluating the function at those samples.
        - It estimates the distribution of possible outputs of a function (f),
          when its inputs are uncertain and described by probability distributions.
 
    args:
        x (list): A list of `UncertainNumber` objects, each representing an input
                 variable with its associated uncertainty.
        f (Callable): A callable function that takes a 1D NumPy array of input
                  values and returns the corresponding output(s). Can be None,
                  in which case only samples are generated.
        results (Propagation_results, optional): An object to store propagation results.
                                            Defaults to None, in which case a new
                                            `Propagation_results` object is created.
        n_sam (int): The number of samples to generate for the chosen sampling method.
                Defaults to 500.
        method (str, optional): The sampling method to use. Choose from:
                            - 'monte_carlo': Monte Carlo sampling (random sampling
                                              from the distributions specified in
                                              the UncertainNumber objects)
                            - 'latin_hypercube': Latin Hypercube sampling (stratified
                                                  sampling for better space coverage)
                            Defaults to 'monte_carlo'.
        save_raw_data (str, optional): Acts as a switch to enable or disable the storage of raw input data when a function (f) 
          is not provided.
          - 'no': Returns an error that no function is provided.
          - 'yes': Returns the full arrays of unique input combinations.

    signature:
        sampling_aleatory_method(x: list, f: Callable, ...) -> propagation_results

    note:
        If the `f` function returns multiple outputs, the code can accomodate. 
        This fuction allows only for the propagation of independent variables.

    returns:
        Returns `Propagation_results` object(s) containing:
            - 'un': UncertainNumber object(s) to represent the empirical distribution(s) of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output values.
                    - 'min' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the minimum of that output.
                    - 'max' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the maximum of that output.
                    - 'bounds' (np.ndarray): 2D array of lower and upper bounds for each output.

    raises:
        ValueError if no function is provided and save_raw_data is 'no' and if invald UP method is selected.

    example:
        # Example usage with different parameters for minimization and maximization

        >>> results = sampling_aleatory_method(x=x, f=f, n_sam = 300, method = 'monte_carlo', save_raw_data = "no")
    """
    if results is None:
        results = Propagation_results()

    if method not in ("monte_carlo", "latin_hypercube"):
        raise ValueError(
            "Invalid sampling method. Choose 'monte_carlo' or 'latin_hypercube'."
        )

    if save_raw_data not in ("yes", "no"):
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")

    print(f"Total number of input combinations for the {method} method: {n_sam}")

    if method == "monte_carlo":
        parameter_samples = np.array([un.random(size=n_sam) for un in x])

    elif method == "latin_hypercube":
        sampler = qmc.LatinHypercube(d=len(x))
        lhd_samples = sampler.random(n=n_sam)

        parameter_samples = []  # Initialize an empty list to store the samples

        for i, un in enumerate(x):  # Iterate over each UncertainNumber in the list 'x'
            # Get the entire column of quantiles for this UncertainNumber
            q_values = lhd_samples[:, i]

            # Now we need to calculate the ppf for each q value in the q_values array
            ppf_values = (
                []
            )  # Initialize an empty list to store the ppf values for this UncertainNumber
            for q in q_values:  # Iterate over each individual q value
                ppf_value = un.ppf(q)  # Calculate the ppf value for this q
                # Add the calculated ppf value to the list
                ppf_values.append(ppf_value)

            # Add the list of ppf values to the main list
            parameter_samples.append(ppf_values)

        # Convert the list of lists to a NumPy array
        parameter_samples = np.array(parameter_samples)

    # Transpose to have each row as a sample
    parameter_samples = parameter_samples.T

    if f is not None:  # Only evaluate if f is provided
        all_output = np.array(
            [f(xi) for xi in tqdm.tqdm(parameter_samples, desc="Evaluating samples")]
        )

        if all_output.ndim == 1:  # If f returns a single output
            # Reshape to a column vector
            all_output = all_output.reshape(-1, 1)

        # if save_raw_data == 'yes':
        results.add_raw_data(x=parameter_samples)
        results.add_raw_data(f=all_output)

    elif save_raw_data == "yes":  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(x=parameter_samples)

    else:
        raise ValueError(
            "No function is provided. Select save_raw_data = 'yes' to save the input combinations."
        )

    return results