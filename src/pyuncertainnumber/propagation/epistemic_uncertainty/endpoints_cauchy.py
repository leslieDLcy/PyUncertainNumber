from __future__ import annotations
from typing import TYPE_CHECKING, Callable

import numpy as np
from rich.progress import track
from pyuncertainnumber.pba.intervals.number import Interval
from scipy.optimize import brentq
from pyuncertainnumber.propagation.utils import Propagation_results, Propagation_rawdata

def cauchydeviates_method(
    x: Interval | np.ndarray,
    f: Callable,
    results: Propagation_results = None,
    analysis_data: Propagation_rawdata = None,
    save_raw_data: bool = False, 
    n_sam: int = 500,
    *,
    xtol: float = 2e-12,
    rtol: float = 8.88e-8,
    maxiter: int = 100,
    min_y0: float = 1e-6,

) -> tuple[Propagation_results, Propagation_rawdata]:
    """
    Propagates interval uncertainty using the approximate Cauchy deviates method.

    Args:
        x (Interval | np.ndarray): A NumPy array where each row is an interval [lower, upper].

        f (Callable): A function that takes a 1D NumPy array of inputs and returns output(s).
        
        results (Propagation_results, optional): An existing results object to populate.
        
        analysis_data (Propagation_rawdata, optional): An existing raw data object to populate.

        save_raw_data (bool, optional): If True and f is None, returns the generated Cauchy
                                      deviate samples and K values. Defaults to False.
        
        n_sam (int, optional): The number of Cauchy deviate samples to generate. Defaults to 500.
        
        xtol (float, optional): Absolute tolerance for the `brentq` root-finding algorithm.
        
        rtol (float, optional): Relative tolerance for the `brentq` root-finding algorithm.
        
        maxiter (int, optional): Maximum iterations for the `brentq` algorithm.
        
        min_y0 (float, optional): The lower bound for the `brentq` root-finding interval.
                                  This is a heuristic and may need adjustment.

    Returns:
        tuple[Propagation_results, Propagation_rawdata]:
            - results: An object containing the final computed interval bounds of the output(s).
            - analysis_data: An object containing raw data, including the generated
              Cauchy samples (`x_samples`, `f_samples`) and K values (`K`).

    Notes:
        
        This method generates samples based on the Cauchy distribution to estimate the
        bounds of the function's output. It is computationally less expensive than
        methods that require optimization but is non-deterministic and provides an
        approximation of the true bounds.
        This method does not track the specific input vectors (`x_min`, `x_max`) that
        produce the minimum and maximum outputs.
        The `brentq` root-finding can sometimes fail if the interval heuristic is
        not suitable for the given function. The function will print a warning and
        continue in such cases.

    raises:
        ValueError: If no function `f` is provided and `save_raw_data` is False.

    example:
        >>> # Define a model function with multiple inputs and outputs
        >>> def myFunctionWithTwoOutputs(x):
        ...     # This function takes a 5-element input vector and returns two outputs.
        ...     i1, i2, i3, i4, i5 = x[0], x[1], x[2], x[3], x[4]
        ...     output1 = i1 + i2 + i3 + i4 + i5
        ...     output2 = i1 * i2 * i3 * i4 * i5
        ...     return output1, output2
        >>>
        >>> # Define input intervals for the 5 variables
        >>> input_intervals = np.array([
        ...     [0.9, 1.1],
        ...     [1.8, 2.2],
        ...     [2.7, 3.3],
        ...     [3.6, 4.4],
        ...     [4.5, 5.5]
        ... ])
        >>>
        >>> # Run the propagation
        >>> results, _ = cauchydeviates_method(
        ...     input_intervals, myFunctionWithTwoOutputs, n_sam=2000
        ... )
        >>>
        >>> # Print the approximate output intervals
        >>> print(results)
    """
    print(f"Total number of function evaluations for Cauchy deviates method: {n_sam}")
    x = np.atleast_2d(x)
    lo, hi = x.T
    xtilde = (lo + hi) / 2
    Delta = (hi - lo) / 2

    # Initialize data containers if not provided
    if analysis_data is None:
        analysis_data = Propagation_rawdata()
    if f is not None:
        # --- ROBUSTNESS FIX: Check output dimension based on size, not type ---
        ytilde = np.atleast_1d(f(xtilde))

        if ytilde.size == 1:  # Handle scalar output
            num_outputs = 1
            ytilde_scalar = ytilde[0] # Use the actual scalar value
            deltaF = np.zeros(n_sam)
            x_samples = np.zeros((n_sam, x.shape[0]))
            K_values = np.zeros(n_sam)

            for k in range(n_sam):
                r = np.random.rand(x.shape[0])
                c = np.tan(np.pi * (r - 0.5))
                K_values[k] = np.max(c)
                delta = Delta * c / K_values[k]
                x_samples[k, :] = xtilde - delta
                # Ensure the function output is treated as a scalar
                f_output_scalar = np.atleast_1d(f(x_samples[k, :]))[0]
                deltaF[k] = K_values[k] * (ytilde_scalar - f_output_scalar)

            def Z(Del):
                return n_sam / 2 - np.sum(1 / (1 + (deltaF / Del) ** 2))

            mask = np.isnan(deltaF)
            filtered_deltaF = deltaF[~mask]

            zRoot = brentq(Z, min_y0, max(filtered_deltaF) / 2, xtol=xtol, rtol=rtol, maxiter=maxiter)
            bounds = np.array([ytilde_scalar - zRoot, ytilde_scalar + zRoot])
            
            if results is None:
                results = Propagation_results(result_type='interval', num_outputs=num_outputs)
            results.add_results(bounds)
            analysis_data.add_sampling_data(x=x_samples, f=None, K=K_values) # f_samples not stored for scalar case
            analysis_data.add_epistemic_data(message="Cauchy method does not track min/max inputs.")

        else:  # Handle array output
            num_outputs = len(ytilde)
            deltaF = np.zeros((n_sam, num_outputs))
            x_samples = np.zeros((n_sam, x.shape[0]))
            f_samples = np.zeros((n_sam, num_outputs))
            K_values = np.zeros(n_sam)

            # Parameter arrays for brentq
            min_y0_arr = np.full(num_outputs, min_y0) if isinstance(min_y0, float) else min_y0
            xtol_arr = np.full(num_outputs, xtol) if isinstance(xtol, float) else xtol
            rtol_arr = np.full(num_outputs, rtol) if isinstance(rtol, float) else rtol
            maxiter_arr = np.full(num_outputs, maxiter) if isinstance(maxiter, int) else maxiter

            for k in track(range(n_sam), description="Calculating Cauchy deviates"):
                r = np.random.rand(x.shape[0])
                c = np.tan(np.pi * (r - 0.5))
                K_values[k] = np.max(c)
                delta = Delta * c / K_values[k]
                x_samples[k, :] = xtilde - delta
                f_samples[k, :] = f(x_samples[k, :])
                for i in range(num_outputs):
                    deltaF[k, i] = K_values[k] * (ytilde[i] - f_samples[k, i])

            bounds_array = np.zeros((num_outputs, 2))
            for i in range(num_outputs):
                mask = np.isnan(deltaF[:, i])
                filtered_deltaF_i = deltaF[:, i][~mask]

                def Z(Del):
                    return n_sam / 2 - np.sum(1 / (1 + (filtered_deltaF_i / Del) ** 2))

                try:
                    zRoot = brentq(Z, min_y0_arr[i], max(filtered_deltaF_i) / 2, xtol=xtol_arr[i], rtol=rtol_arr[i], maxiter=maxiter_arr[i])
                except ValueError:
                    print(f"Warning: brentq failed for output {i}. Using 0 for zRoot.")
                    zRoot = 0
                bounds_array[i, :] = [ytilde[i] - zRoot, ytilde[i] + zRoot]

            if results is None:
                results = Propagation_results(result_type='interval', num_outputs=num_outputs)
            results.add_results(bounds_array)
            analysis_data.add_sampling_data(x=x_samples, f=f_samples, K=K_values)
            analysis_data.x_central = xtilde # Also save central point here
            analysis_data.add_epistemic_data(message="Cauchy method does not track min/max inputs.")

    elif save_raw_data:
        if results is None: results = Propagation_results(result_type='interval')
        x_samples, K_values = np.zeros((n_sam, x.shape[0])), np.zeros(n_sam)
        for k in range(n_sam):
            r = np.random.rand(x.shape[0])
            c = np.tan(np.pi * (r - 0.5))
            K_values[k] = np.max(c)
            x_samples[k] = xtilde - (Delta * c / K_values[k])
        analysis_data.add_sampling_data(x=x_samples, f=None, K=K_values)
        analysis_data.x_central = xtilde  # Also save central point here
    else:
        raise ValueError("No function `f` was provided. Set save_raw_data=True to get the input combinations.")

    return results, analysis_data