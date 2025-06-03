import numpy as np
from rich.progress import track
from typing import Callable
from scipy.stats import qmc  # Import Latin Hypercube Sampling from SciPy
from pyuncertainnumber.propagation.utils import Propagation_results
from pyuncertainnumber import UncertainNumber
from ...pba.distributions import Distribution


def sampling_aleatory_method(
    xs: list[Distribution],
    f: Callable,
    method: str = "monte_carlo",
    n_sam: int = 1_000,
    results: Propagation_results = None,
    save_raw_data=False,
) -> Propagation_results:  # Specify return type
    #! leslie: I prefer returning the raw output sample
    """Performs aleatory uncertainty propagation using Monte Carlo or Latin Hypercube sampling,
        when its inputs are uncertain and described by probability distributions.

    args:
        xs (list): A list of `UncertainNumber` or `Distribution` objects;
        f (Callable): A callable function that takes input
                  values and returns the corresponding output(s). Can be None,
                  in which case only samples are generated.
        results (Propagation_results, optional): An object to store propagation results.
                                            Defaults to None, in which case a new
                                            `Propagation_results` object is created.
        method (str, optional): The sampling method to use. Choose from:
                            - 'monte_carlo': Monte Carlo sampling (random sampling
                                              from the distributions specified in
                                              the UncertainNumber objects)
                            - 'latin_hypercube': Latin Hypercube sampling (stratified
                                                  sampling for better space coverage)
                            Defaults to 'monte_carlo'.
        n_sam (int): The number of samples to generate for the chosen sampling method.
                Defaults to 1000.
        save_raw_data (str, optional): Acts as a switch to enable or disable the storage of raw
          input data when a function (f) is not provided.
          - 'no': Returns an error that no function is provided.
          - 'yes': Returns the full arrays of unique input combinations.

    signature:
        sampling_aleatory_method(x: list, f: Callable, ...) -> propagation_results

    note:
        If the `f` function returns multiple outputs, the code can accomodate.
        This fuction allows only for the propagation of independent variables.

    returns:
        Returns response samples or a `Propagation_results` object:

        `Propagation_results` object(s) containing:
            - 'un': UncertainNumber object(s) to represent the empirical distribution(s) of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output values.

    raises:
        ValueError if no function is provided and save_raw_data is 'no' and if invald UP method is selected.

    example:
        # Example usage with different parameters for minimization and maximization

        >>> results = sampling_aleatory_method(x=x, f=f, n_sam = 300, method = 'monte_carlo', save_raw_data = "no")
    """
    if results is None:
        results = Propagation_results()

    assert callable(f), "The function f must be a callable."

    print(f"Total number of input combinations for the {method} method: {n_sam}")

    match method:
        case "monte_carlo":
            if isinstance(xs[0], Distribution):
                samples = [d.sample(n_sam) for d in xs]
            elif isinstance(xs[0], UncertainNumber):
                samples = [un.construct.sample(n_sam) for un in xs]
            else:
                raise ValueError(
                    "The input must be a list of UncertainNumber or Distribution objects."
                )
            # samples = np.stack(samples, axis=1)
            # samples = np.array([un.random(size=n_sam) for un in x])
        case "latin_hypercube":
            sampler = qmc.LatinHypercube(d=len(xs))
            lhs_samples = sampler.random(n=n_sam)  # u-space (n, d)

            samples = np.concatenate(
                [d.alpha_cut(lhs_samples[:, i]) for i, d in enumerate(xs)], axis=1
            )
        case _:
            raise ValueError("Invalid UP method!")

    # TODO discuss the logic to save verbose raw data @ioannaioan
    # TODO seems to be different behaviors between aleatory and epistemic methods

    if save_raw_data:
        # Transpose to have each row as a sample
        samples = samples.T

        if f is not None:  # Only evaluate if f is provided
            all_output = np.array(
                [
                    f(xi)
                    for xi in track(
                        samples, description="function evaluations for samples"
                    )
                ]
            )

            if all_output.ndim == 1:  # If f returns a single output
                # Reshape to a column vector
                all_output = all_output.reshape(-1, 1)

            # if save_raw_data == 'yes':
            results.add_raw_data(x=samples)
            results.add_raw_data(f=all_output)
        else:
            raise ValueError(
                "No function is provided. Select save_raw_data = 'yes' to save the input combinations."
            )

        return results
    else:
        # simpler logic
        return f(*samples)
