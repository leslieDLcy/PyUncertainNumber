from pathlib import Path
import csv
import pandas as pd
import numpy as np
from typing import Callable
from pyuncertainnumber.characterisation.uncertainNumber import (
    UncertainNumber,
    Distribution,
)
# TODO Not quite sure if data saving behavior will be consistent across all aleatory, epistemic, mixed methods
# TODO what will be saved for aleatory case? what will be saved for mixed case?


class Propagation_rawdata:
    """
    A data container for the raw inputs, outputs, and intermediate results
    from an uncertainty propagation analysis. This class stores the detailed
    data from sampling or optimization methods.
    """
    def __init__(self):
        """Initializes all raw data fields to their default empty states."""
        # --- Data from sampling-based methods ---
        self.x_samples = None  # Input samples
        self.f_samples = None  # Corresponding output samples from the model
        self.x_central = None
        self.f_central = None

        # --- Data from optimization/epistemic methods ---
        self.x_min = None  # Input vectors that yield the minimum for each output
        self.x_max = None  # Input vectors that yield the maximum for each output
        self.opti_message = None # Success/failure message from the optimizer
        self.opti_iterations = None # Total iterations for optimization

        # --- Data for specific methods ---
        self.K = None      # e.g., Stiffness matrix for Cauchy analysis
        self.sign_x = None # Sensitivity signs for the Extreme Points method

    def add_sampling_data(self, x, f, K=None):
        """
        Adds data from a sampling-based analysis (e.g., Monte Carlo).

        Args:
            x (np.ndarray): The input sample matrix.
            f (np.ndarray): The corresponding output sample matrix.
            K (any, optional): Additional data to store (e.g., stiffness matrix).
        """
        self.x_samples = x
        self.f_samples = f
        if K is not None:
            self.K = K

    def add_epistemic_data(self, x_min=None, x_max=None, sign_x=None,
                             message=None, iterations=None):
        """
        Adds data from an epistemic analysis (e.g., optimization, extreme points).

        Args:
            x_min (np.ndarray, optional): Input(s) yielding the minimum output(s).
            x_max (np.ndarray, optional): Input(s) yielding the maximum output(s).
            sign_x (np.ndarray, optional): Sensitivity signs for extreme points method.
            message (list/str, optional): Optimizer success/failure messages.
            iterations (list/int, optional): Optimizer iteration counts.
        """
        if x_min is not None:
            self.x_min = x_min
        if x_max is not None:
            self.x_max = x_max
        if sign_x is not None:
            self.sign_x = sign_x
        if message is not None:
            self.opti_message = message
        if iterations is not None:
            self.opti_iterations = iterations


class Propagation_results:
    """
    Stores the final results of an uncertainty propagation analysis.
    The structure of the output depends on the type of uncertainty.

    Args:
        result_type (str): The type of result. Must be one of
                           ['interval', 'pbox', 'aleatory'].
        num_outputs (int): The number of outputs from the model. Defaults to 1.
    """

    def __init__(self, result_type: str, num_outputs: int = 1):
        if result_type not in ['interval', 'pbox', 'aleatory']:
            raise ValueError("result_type must be 'interval', 'pbox', or 'aleatory'")

        self.result_type = result_type
        self.num_outputs = num_outputs
        self.data = {}  # Dictionary to hold the final, structured results
        #self.raw_data = PropagationRawData()

    def add_results(self, data):
        """
        Adds the final computed results, structuring them based on result_type.

        Args:
            data: The final results to be stored.
                  - For 'interval': A (num_outputs, 2) numpy array of [min, max] bounds.
                  - For 'pbox': A tuple of (left_data, right_data) arrays.
                  - For 'aleatory': An array of output samples.
        """
        if self.result_type == 'interval':
            self.data['bounds'] = data
        elif self.result_type == 'pbox':
            self.data['left_data'], self.data['right_data'] = data
        elif self.result_type == 'aleatory':
            self.data['samples'] = data
# class Propagation_results:
#     """Stores the results of uncertainty propagation with multiple outputs, sharing raw_data x and f.

#     args:
#         un (np.ndarray): np.array of UncertainNumber objects (one for each output).
#         raw_data (dict): Dictionary containing raw data shared across outputs:
#             x (np.ndarray): Input values.
#             f (np.ndarray): Output values.
#             min (np.ndarray): Array of dictionaries, one for each output,
#                               containing 'x', 'f' for the minimum of that output.
#             max (np.ndarray): Array of dictionaries, one for each output,
#                               containing 'x', 'f' for the maximum of that output.
#             bounds (np.ndarray): 2D array of lower and upper bounds for each output.

#     notes:
#         - use `foo.un` to access the UncertainNumber objects.
#     """

#     def __init__(self, un: np.ndarray = None, raw_data: dict = None):
#         if un is None:
#             # Initialize as an empty array if un is None
#             self.un = np.array([])
#         else:
#             self.un = un

#         if raw_data is not None:
#             self.raw_data = raw_data
#         else:
#             self.raw_data = {
#                 "x": None,
#                 "f": None,
#                 "min": np.array([]),
#                 "max": np.array([]),
#                 "bounds": np.array([]),
#             }

#     def add_raw_data(self, x=None, f=None, K=None, sign_x: np.ndarray = None):
#         """Adds raw data to the results."""
#         if x is not None:
#             self.raw_data["x"] = x
#         if K is not None:
#             self.raw_data["K"] = K
#         if f is not None:
#             self.raw_data["f"] = f
#         if sign_x is not None:
#             if (
#                 isinstance(sign_x, np.ndarray) and len(sign_x.shape) > 1
#             ):  # Multiple outputs
#                 self.raw_data["sign_x"] = sign_x
#             else:  # Single output
#                 self.raw_data["sign_x"] = np.array([sign_x])  # Wrap in an array

#     def summary(self):
#         """Prints the results in a formatted way, handling None values and multiple outputs."""

#         if self.raw_data["f"] is not None:  # Check if 'f' exists and is not None
#             if len(self.raw_data["f"].shape) == 1:  # 1D array, single output
#                 num_outputs = 1
#             else:  # 2D array, multiple outputs
#                 num_outputs = self.raw_data["f"].shape[1]  # Number of columns
#         # Check if 'bounds' exists and is not None (Corrected)
#         elif self.raw_data["bounds"] is not None and len(self.raw_data["bounds"]) > 0:
#             if len(self.raw_data["bounds"].shape) == 3:  # 1D array, single output
#                 num_outputs = 1
#             else:  # 2D array, multiple outputs
#                 # Number of rows (Corrected)
#                 num_outputs = self.raw_data["bounds"].shape[0]
#         else:
#             num_outputs = 1  # Or handle the case where 'f' is None appropriately

#         for i in range(num_outputs):
#             print("-" * 30)
#             print(f"Output {i+1}:")
#             print("-" * 30)

#             if self.un is not None:
#                 if num_outputs == 1:
#                     print("Uncertain Number:", self.un)
#                 else:
#                     print("Uncertain Number:", self.un[i])
#             else:
#                 print("Uncertain Number: None")

#             if (
#                 "bounds" in self.raw_data
#                 and self.raw_data["bounds"] is not None
#                 and len(self.raw_data["bounds"]) > 0
#             ):
#                 print("-" * 30)
#                 if num_outputs == 1:
#                     print("Bounds:", self.raw_data["bounds"])
#                 else:
#                     print("Bounds:", self.raw_data["bounds"][i])

#             if (
#                 "min" in self.raw_data
#                 and self.raw_data["min"] is not None
#                 and len(self.raw_data["min"]) > 0
#             ):
#                 print("-" * 30)
#                 print("Minimum:")
#                 min_data = self.raw_data["min"][i]
#                 if min_data.get("f") is not None:
#                     print("f:", min_data.get("f"))
#                     if min_data.get("x") is None:  # Handle the case where 'x' is None
#                         print("x: None")
#                     else:
#                         print("x:", min_data.get("x"))
#                     # Print additional results only for local_optimisation
#                     if "final_simplex" in min_data:
#                         print("niterations:", min_data.get("niterations"))
#                         print("nfevaluations:", min_data.get("nfevaluations"))
#                         print("final_simplex:", min_data.get("final_simplex"))
#                         print("message:", min_data.get("message"))
#                     if "ngenerations" in min_data:
#                         print("niterations:", min_data.get("niterations"))
#                         print("ngenerations", min_data.get("ngenerations"))
#                         print("message:", min_data.get("message"))

#             if (
#                 "max" in self.raw_data
#                 and self.raw_data["max"] is not None
#                 and len(self.raw_data["max"]) > 0
#             ):
#                 print("-" * 30)
#                 print("Maximum:")
#                 max_data = self.raw_data["max"][i]
#                 if max_data.get("f") is not None:
#                     print("f:", max_data.get("f"))
#                     if max_data.get("x") is None:  # Handle the case where 'x' is None
#                         print("x: None")
#                     else:
#                         print("x:", max_data.get("x"))
#                     if "final_simplex" in max_data:
#                         print("niterations:", max_data.get("niterations"))
#                         print("nfevaluations:", max_data.get("nfevaluations"))
#                         print("final_simplex:", max_data.get("final_simplex"))
#                         print("message:", max_data.get("message"))
#                     if "ngenerations" in max_data:
#                         print("niterations:", max_data.get("niterations"))
#                         print("ngenerations:", max_data.get("ngenerations"))
#                         print("message:", max_data.get("message"))

#             if "sign_x" in self.raw_data:
#                 print("-" * 30)
#                 print("sign_x:", self.raw_data["sign_x"][i])

#         print("-" * 30)
#         print("Input combinations and corresponding output(s):")
#         # Check if 'x' is not None
#         if self.raw_data["x"] is not None and len(self.raw_data["x"]) > 0:
#             print("x:", self.raw_data["x"])
#         else:
#             print("x: None")

#         if "K" in self.raw_data:  # Check if the keys exist
#             print("-" * 30)
#             print("K:", self.raw_data["K"])
#         print("-" * 30)

#         # Check if 'x' is not None
#         if self.raw_data["f"] is not None and len(self.raw_data["f"]) > 0:
#             print("f:", self.raw_data["f"])  # Print directly if single output
#         else:
#             print("f: None")
#         print("-" * 30)


def process_alea_results(results,save_raw_data: bool = False, base_path: str = None, method: str = None, fun=None):
    """
    args:
        - results (Propagation_results): A `Propagation_results` object containing raw
                                epistemic propagation results. This object is
                                modified in-place.

    signature:
        - process_alea_results(results: Propagation_results) -> Propagation_results

    notes:
        - Processes the results of aleatory uncertainty propagation.

        - This function takes a `Propagation_results` object containing raw aleatory
            propagation results and performs the following actions:

            1. Creates `Distribution` objects:
                - If output data exists in `results.raw_data['f']`, it creates an 'UncertainNumber'
                    object  for each output dimension using the sample data.
                - These `UncertainNumber` objects are stored in `results.un`.
                - They have essense = 'distribution'

            2. Saves raw data (optional):
                - If `save_raw_data` is set to 'yes', it saves the raw propagation data
                    (input samples and corresponding output values) to a file.

    returns:
        - Propagation_results: The modified `Propagation_results` object with
                        `UncertainNumber` objects added to `results.un` and
                        potentially with raw data saved to a file.

    raises:
        - ValueError: If the shape of `results.raw_data['f']` is invalid

    examples:
        >>> a = mixed_propagation(vars= [y, L, I, F, E],
        >>>                 fun= cantilever_beam_func,
        >>>                 method= 'monte_carlo',
        >>>                 n_disc=8,
        >>>                 save_raw_data= "no"
        >>>             )
    """
    if results.raw_data["f"] is None:  # Access raw_data from results object
        # UncertainNumber(essence="distribution", distribution_parameters=None, **kwargs)
        results.un = None
    else:
        results.un = []
        # Access raw_data from results object
        for sample_data in results.raw_data["f"].T:
            # results.un.append(UncertainNumber(essence="distribution", distribution_parameters=sample_data, **kwargs))
            results.un.append(Distribution(sample_data=sample_data))

    if save_raw_data == "yes":
        res_path = create_folder(base_path, method)
        save_results(results.raw_data, method=method, res_path=res_path, fun=fun)

    return results


def process_results(
    results: Propagation_results,
    analysis_data: Propagation_rawdata,
    *,
    save_raw_data: bool = False,
    base_path: str = None,
    method: str = None,
    fun: Callable = None,
) -> Propagation_results:
    """
    Processes propagation results to create UncertainNumber objects and optionally saves raw data.

    Args:
        results (PropagationResults): The object containing final results (e.g., bounds).
                                      This object is modified in-place by adding a '.un' attribute.
        analysis_data (PropagationRawData): The object containing all intermediate data.
        save_raw_data (bool, optional): If True, saves the `analysis_data` object to a file.
        base_path (str, optional): The base directory for saving raw data.
        method (str, optional): The name of the method, used for the save folder.
        fun (Callable, optional): The function analyzed, used for the save filename.

    Returns:
        PropagationResults: The modified `results` object, now containing `UncertainNumber`
                            objects in its `.un` attribute.
    """
    if results.result_type != 'interval':
        print(f"Processing skipped: result type is '{results.result_type}', not 'interval'.")
        return results

    bounds_data = results.data.get('bounds')

    if bounds_data is None or bounds_data.size == 0:
        results.un = UncertainNumber(essence="interval", bounds=[-np.inf,np.inf])
    else:
        if bounds_data.ndim == 2:  # Multi-output case
            results.un = [UncertainNumber(essence="interval", bounds=bound) for bound in bounds_data]
        elif bounds_data.ndim == 1 and len(bounds_data) == 2:  # Single-output case
            results.un = UncertainNumber(essence="interval", bounds=bounds_data)
        else:
            raise ValueError("Invalid shape for 'bounds' in results.data. Expected 2D array or 1D array with two values.")

    if save_raw_data == True:
         # Create the folder for the results
        res_path = create_folder(base_path, method)
        print('res_path:', res_path)

        # Call save_results once. It will handle the post-processing and saving.
        # It also returns the processed DataFrame if you need to use it.
        if save_raw_data:
            if base_path is None or method is None:
                print("Warning: save_raw_data is True, but 'base_path' or 'method' was not provided. Skipping save.")
            else:
                res_path = create_folder(base_path, method)
                save_results(analysis_data, method=method, res_path=res_path)
       
    return results

# def process_results(results: Propagation_results,save_raw_data: bool = False, base_path: str = np.nan, method: str = None, fun=None):
#     """
#     args:
#         - results (Propagation_results): A `Propagation_results` object containing raw
#                                 epistemic propagation results. This object is
#                                 modified in-place.

#     notes:
#         - Processes the results of epistemic uncertainty propagation.

#         - This function takes a `Propagation_results` object containing raw epistemic
#             propagation results and performs the following actions:

#             1. Creates `UncertainNumber` objects:
#                 - If output bounds exist in `results.raw_data['bounds']`, it creates
#                     `UncertainNumber` objects with "interval" essence, representing the
#                     resulting interval uncertainty.
#                 - It handles both single-output (1D array of bounds) and multi-output
#                     (2D array of bounds) cases.
#                 - These `UncertainNumber` objects are stored in `results.un`.

#             2. Saves raw data (optional):
#                 - If `save_raw_data` is set to 'yes', it saves the raw propagation data
#                     to a file.

#     signature:
#         - process_results(results: Propagation_results) -> Propagation_results

#     returns:
#         - Propagation_results: The modified `Propagation_results` object with
#                         `UncertainNumber` objects added to `results.un` and
#                         potentially with raw data saved to a file.

#     raises:
#         - ValueError: If the shape of `results.raw_data['bounds']` is invalid.

#     """
#     print("bounds array:", results.raw_data["bounds"])
#     for bound in results.raw_data["bounds"]:
#         print("bound:", bound, "type:", type(bound), "shape:", np.shape(bound))
#         UncertainNumber(essence="interval", bounds=bound)

#     if results.raw_data["bounds"] is None or results.raw_data["bounds"].size == 0:
#         results.un = UncertainNumber(essence="interval", bounds=None)
#     else:
       

#         if results.raw_data["bounds"].ndim == 2:  # 2D array
#             results.un = [
#                 UncertainNumber(essence="interval", bounds=bound)
#                 for bound in results.raw_data["bounds"]
#             ]
#         # 1D array
#         elif (
#             results.raw_data["bounds"].ndim == 1
#             and len(results.raw_data["bounds"]) == 2
#         ):
#             results.un = UncertainNumber(
#                 essence="interval", bounds=results.raw_data["bounds"]
#             )
#         else:
#             raise ValueError(
#                 "Invalid shape for 'bounds'. Expected 2D array or 1D array with two values."
#             )

#     if save_raw_data == True:
#         res_path = create_folder(base_path, method)
#         save_results(results.raw_data, method=method, res_path=res_path, fun=fun)

#     return results

    def process_mixed_results(results: Propagation_results):
        """
        args:
            - results (Propagation_results): A `Propagation_results` object containing raw
                                    epistemic propagation results. This object is
                                    modified in-place.

        signature:
            - process_mixed_results(results: Propagation_results) -> Propagation_results

        notes:
            - Processes the results of mixed uncertainty propagation.

            - This function takes a `Propagation_results` object containing raw aleatory
              propagation results and performs the following actions:

                1. Creates `UncertainNumber` objects:
                    - If output data exists in `results.raw_data['bounds']`, it creates an 'UncertainNumber'
                      object  for each output dimension using the sample data.
                    - These `UncertainNumber` objects are stored in `results.un`.
                    - The `UncertainNumber` has essense = 'pbox'.

                2. Saves raw data (optional):
                    - If `save_raw_data` is set to 'yes', it saves the raw propagation data
                      (input samples and corresponding output values) to a file.

        returns:
            - Propagation_results: The modified `Propagation_results` object with
                          `UncertainNumber` objects added to `results.un` and
                          potentially with raw data saved to a file.

        """
        # if results.raw_data['bounds'] is None or results.raw_data['bounds'].size == 0:
        #     results.un = None
        # else:
        #     if results.raw_data['bounds'].ndim == 4:  # 2D array
        #         results.un = UncertainNumber( essence='interval', bounds=[1, 2])  #[UncertainNumber(essence="pbox", pbox_parameters = bound, **kwargs) for bound in results.raw_data['bounds']]
        #     elif results.raw_data['bounds'].ndim == 3:  # 1D array
        #         results.un =  UncertainNumber( essence='interval', bounds=[1, 2]) #UncertainNumber(essence="pbox",  pbox_parameters=results.raw_data['bounds'], **kwargs)
        #     else:
        #         raise ValueError("Invalid shape for 'bounds'. Expected 2D array or 1D array with two values.")

        # if save_raw_data == "yes":
        # res_path = create_folder(base_path, method)
        # save_results(results.raw_data, method=method, res_path=res_path, fun=fun)

        return results


# * ------------- more functions ------------- *#

def header_results(analysis_data: Propagation_rawdata) -> list[str]:
    """
    Determines a generic header for the raw data DataFrame by checking
    which data attributes are present in the analysis_data object.

    Args:
        analysis_data (Propagation_rawdata): The object containing the raw data.

    Returns:
        list[str]: A list of strings for the DataFrame header.
    """
    header = []
    # Add headers for outputs (y0, y1, ...)
    if analysis_data.f_samples is not None:
        num_outputs = 1 if analysis_data.f_samples.ndim == 1 else analysis_data.f_samples.shape[1]
        header.extend([f"y{i}" for i in range(num_outputs)])

    # Add headers for inputs (x0, x1, ...)
    if analysis_data.x_samples is not None:
        num_inputs = analysis_data.x_samples.shape[1]
        header.extend([f"x{i}" for i in range(num_inputs)])

    # Add header for Cauchy's K value if it exists
    if analysis_data.K is not None:
        header.append("K")
    
    # Add headers for sign_x if it exists (e.g., from extreme points)
    if analysis_data.sign_x is not None:
        num_signs = 1 if analysis_data.sign_x.ndim == 1 else analysis_data.sign_x.shape[1]
        header.extend([f"sign_x{i}" for i in range(num_signs)])

    return header


def post_processing(analysis_data: Propagation_rawdata, res_path: str | Path = None) -> pd.DataFrame:
    """
    Dynamically combines all available raw data into a DataFrame and optionally saves it.
    """
    dfs_to_concat = []

    # Add output samples if they exist
    if analysis_data.f_samples is not None:
        dfs_to_concat.append(pd.DataFrame(analysis_data.f_samples))

    # Add input samples if they exist
    if analysis_data.x_samples is not None:
        dfs_to_concat.append(pd.DataFrame(analysis_data.x_samples))
    
    # Add Cauchy's K value if it exists
    if analysis_data.K is not None:
        dfs_to_concat.append(pd.DataFrame(analysis_data.K, columns=['K']))

    # Add sign_x data if it exists
    if analysis_data.sign_x is not None:
        num_signs = 1 if analysis_data.sign_x.ndim == 1 else analysis_data.sign_x.shape[1]
        sign_headers = [f"sign_x{i}" for i in range(num_signs)]
        dfs_to_concat.append(pd.DataFrame(analysis_data.sign_x, columns=sign_headers))

    if not dfs_to_concat:
        print("No raw data found in analysis_data to process.")
        return pd.DataFrame()

    # Concatenate all available dataframes horizontally
    df_combined = pd.concat(dfs_to_concat, axis=1)
    
    # Generate the header dynamically based on the available data
    df_combined.columns = header_results(analysis_data)

    if res_path:
        # Save the main raw data and NaN log
        create_csv(res_path, "Raw_data.csv", df_combined)
        if analysis_data.f_samples is not None and df_combined.isnull().values.any():
            create_csv(res_path, "NA_log.csv", df_combined[df_combined.isnull().any(axis=1)])

        # Save min/max x-values if they exist
        if analysis_data.x_min is not None and analysis_data.x_max is not None:
            min_max_records = []
            num_inputs = analysis_data.x_min[0].shape[1]
            x_headers = [f"x{i}" for i in range(num_inputs)]
            for i, x_vectors in enumerate(analysis_data.x_min):
                for vec in x_vectors:
                    min_max_records.append({'output_idx': i, 'bound_type': 'min', **dict(zip(x_headers, vec))})
            for i, x_vectors in enumerate(analysis_data.x_max):
                for vec in x_vectors:
                    min_max_records.append({'output_idx': i, 'bound_type': 'max', **dict(zip(x_headers, vec))})
            if min_max_records:
                create_csv(res_path, "min_max_inputs.csv", pd.DataFrame(min_max_records))
        
        if analysis_data.x_central is not None and analysis_data.f_central is not None:
            central_data = {
                **{f'x_central_{i}': val for i, val in enumerate(np.atleast_1d(analysis_data.x_central))},
                **{f'f_central_{i}': val for i, val in enumerate(np.atleast_1d(analysis_data.f_central))}
            }
            df_central = pd.DataFrame([central_data])
            create_csv(res_path, "central_point_data.csv", df_central)

    return df_combined


def create_folder(base_path: str | Path, folder_name: str) -> Path:
    """
    Creates a folder at the specified path.

    Args:
        base_path (str | Path): The base directory.
        folder_name (str): The name of the new folder to create.

    Returns:
        Path: The path to the created folder.
    """
    res_path = Path(base_path) / folder_name
    res_path.mkdir(parents=True, exist_ok=True)
    return res_path


def create_csv(res_path: str | Path, filename: str, data: pd.DataFrame):
    """
    Saves a pandas DataFrame to a CSV file.

    Args:
        res_path (str | Path): The directory path for saving the file.
        filename (str): The name of the CSV file.
        data (pd.DataFrame): The DataFrame to save.
    """
    file_path = Path(res_path) / filename
    data.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def save_results(
    analysis_data: Propagation_rawdata,
    method: str,
    res_path: str | Path,
) -> pd.DataFrame:
    """
    A wrapper function to post-process and save raw data from an analysis.

    Args:
        analysis_data (PropagationRawData): The object containing the raw data.
        method (str): The name of the propagation method used.
        res_path (str | Path): The directory path for saving.

    Returns:
        pd.DataFrame: The processed DataFrame containing the combined raw data.
    """
    print(f"\n--- Post-processing results for method: '{method}' ---")
    df = post_processing(analysis_data, res_path=res_path)
    return df
# def header_results(analysis_data: Propagation_rawdata) -> list[str]:
#     """
#     Determines a generic header for the raw data DataFrame.

#     Args:
#         analysis_data (PropagationRawData): The object containing the raw data.

#     Returns:
#         list[str]: A list of strings for the DataFrame header.
#     """
#     # Create header for outputs (y0, y1, ...)
#     header_y = []
#     if analysis_data.f_samples is not None:
#         num_outputs = 1 if analysis_data.f_samples.ndim == 1 else analysis_data.f_samples.shape[1]
#         header_y = [f"y{i}" for i in range(num_outputs)]

#     # Create header for inputs (x0, x1, ...)
#     header_x = []
#     if analysis_data.x_samples is not None:
#         num_inputs = analysis_data.x_samples.shape[1]
#         header_x = [f"x{i}" for i in range(num_inputs)]

#     # Special handling for 'K' if it exists (e.g., in Cauchy method)
#     if analysis_data.K is not None:
#         header_x.append("K")

#     return header_y + header_x


# def post_processing(
#     analysis_data: Propagation_rawdata, res_path: str | Path = None
# ) -> pd.DataFrame:
#     """
#     Post-processes the raw data from an uncertainty propagation analysis.

#     This function combines the raw input and output samples into a pandas
#     DataFrame, checks for NaN values, and optionally saves the DataFrame to a CSV file.

#     Args:
#         analysis_data (PropagationRawData): An object containing the raw input and output samples.
#         res_path (str | Path, optional): The directory path to save results. Defaults to None.

#     Returns:
#         pd.DataFrame: A DataFrame containing the combined raw data.
#     """
#     if analysis_data.x_samples is None:
#         print("No input samples found in analysis_data.")
#         return pd.DataFrame()

#     all_input = analysis_data.x_samples
#     all_output = analysis_data.f_samples

#     # Combine input and output data into a single DataFrame
#     if all_output is None:
#         print("No function was evaluated. DataFrame contains inputs only.")
#         df_combined = pd.DataFrame(all_input)
#     else:
#         # For Cauchy, K is often separate; here we assume it's part of x_samples if needed
#         df_input = pd.DataFrame(all_input)
#         df_output = pd.DataFrame(all_output)
#         df_combined = pd.concat([df_output, df_input], axis=1)

#     # Set the column headers
#     df_combined.columns = header_results(analysis_data)

#     if res_path:
#         create_csv(res_path, "Raw_data.csv", df_combined)

#     # Check for NaN values in the output columns if they exist
#     if all_output is not None:
#         if df_combined.isnull().values.any():
#             df_NA = df_combined[df_combined.isnull().any(axis=1)]
#             print(f"Found {len(df_NA)} rows with NaN values.")
#             if res_path:
#                 create_csv(res_path, "NA_log.csv", df_NA)
#         else:
#             print("No NA values found in the output.")

#     return df_combined


# def create_folder(base_path: str | Path, folder_name: str) -> Path:
#     """
#     Creates a folder at the specified path.

#     Args:
#         base_path (str | Path): The base directory.
#         folder_name (str): The name of the new folder to create.

#     Returns:
#         Path: The path to the created folder.
#     """
#     res_path = Path(base_path) / folder_name
#     res_path.mkdir(parents=True, exist_ok=True)
#     return res_path


# def create_csv(res_path: str | Path, filename: str, data: pd.DataFrame):
#     """
#     Saves a pandas DataFrame to a CSV file.

#     Args:
#         res_path (str | Path): The directory path for saving the file.
#         filename (str): The name of the CSV file.
#         data (pd.DataFrame): The DataFrame to save.
#     """
#     file_path = Path(res_path) / filename
#     data.to_csv(file_path, index=False)
#     print(f"Data saved to {file_path}")


# def save_results(
#     analysis_data: Propagation_rawdata,
#     method: str,
#     res_path: str | Path,
#     fun: Callable = None,
# ):
#     """
#     A wrapper function to post-process and save raw data from an analysis.

#     Args:
#         analysis_data (PropagationRawData): The object containing the raw data.
#         method (str): The name of the propagation method used.
#         res_path (str | Path): The directory path for saving.
#         fun (Callable, optional): The function that was analyzed. Defaults to None.
#     """
#     # The main logic is now handled by post_processing
#     print(f"\n--- Post-processing results for method: '{method}' ---")
#     post_processing(analysis_data, res_path=res_path)


# def header_results(all_output, all_input, method=None):
#     """
#     Determine generic header for output and input.

#     Args:
#         all_output (np.ndarray): A NumPy array containing the output values.
#         all_input (np.ndarray): A NumPy array containing the input values.

#     Returns:
#         list: A list of strings representing the header for the combined DataFrame.
#     """
#     if all_output is None:
#         header_y = []
#     else:
#         if all_output.ndim == 1:
#             len_y = 1
#         else:
#             len_y = all_output.shape[1]
#         header_y = ["y" + str(i) for i in range(len_y)]

#     if method in "cauchy":
#         m = all_input.shape[1] - 1  # Exclude 'K' from the count
#         header_x = ["x" + str(i) for i in range(m)] + ["K"]  # Add 'K' at the end
#     else:
#         m = all_input.shape[1]
#         header_x = ["x" + str(i) for i in range(m)]

#     header = header_y + header_x
#     return header


# def post_processing(
#     all_input: np.ndarray, all_output: np.ndarray = None, method=None, res_path=None
# ):
#     """Post-processes the results of an uncertainty propagation (UP) method.

#     This function takes the input and output values from a UP method, combines them into a
#     pandas DataFrame, and optionally saves the raw data to a CSV file. It also checks for
#     NaN values in the output and logs them with their corresponding input values if found.
#     If all_output is None, it creates a DataFrame with only the input data.

#     Args:
#         all_input (np.ndarray): A NumPy array containing the input values used in the UP method.
#         all_output (np.ndarray, optional): A NumPy array containing the corresponding output
#                                             values from the UP method. Defaults to None.
#         res_path (str, optional): The path to the directory where the results will be saved.
#                                     Defaults to None.

#     Returns:
#         pandas.DataFrame: A pandas DataFrame containing the combined output and input data
#                         (if all_output is provided). If all_output is None, it returns a
#                         DataFrame with only the input data.
#     """

#     if all_output is None:
#         print("No function was evaluated. Only input is available")

#         df_input = pd.DataFrame(all_input)

#         # Handle Cauchy input with 'x' and 'K' (moved outside the if block)
#         # if method in ( "endpoints_cauchy"):
#         #     x_fields = pd.DataFrame(all_input)  # Get all 'x' field names
#         #     print("x_fields", x_fields)
#         #     x_data = all_input[x_fields]  # Select only the 'x' fields
#         #     df_input = pd.DataFrame(x_data)
#         # else:
#         #     df_input = pd.DataFrame(all_input)

#         header = header_results(all_output=None, all_input=all_input, method=method)
#         df_input.columns = header
#         df_output_input = df_input

#     else:
#         # Transform np.array input-output into pandas data.frame
#         df_input = pd.DataFrame(all_input)
#         df_output = pd.DataFrame(all_output)

#         # Create a single output input data.frame
#         df_output_input = pd.concat([df_output, df_input], axis=1)

#         # determine generic header for output and input
#         header = header_results(all_output, all_input, method)
#         df_output_input.columns = header

#     # Return .csv with raw data only if asked ###
#     if res_path is not None:
#         create_csv(res_path, "Raw_data.csv", df_output_input)

#     # Check for NaN values ONLY if all_output is provided
#     if all_output is not None:
#         df_NA = df_output_input[df_output_input.isna().any(axis=1)]
#         if len(df_NA) != 0:
#             # The input values are rounded to ensure equality
#             df_NA = df_NA.apply(np.round, args=[4])
#             df_NA_unique = df_NA.drop_duplicates(keep="first", ignore_index=True)

#             create_csv(res_path, "NAlog.csv", df_NA_unique)
#         else:
#             print("There are no NA values produced by the input")

#     return df_output_input


# def create_folder(base_path, method):
#     """Creates a folder named after the called UP method where the results files are stored

#     args:
#         - base_path: The base path
#         - method: the name of the called method

#     signature:
#         create_folder(base_path: string, method: string ) -> path.folder

#     note:
#         - the augument `base_path` will specify the location of the created results folder.
#         - the argument `method` will provide the name for the results folder.

#     return:
#         -  A folder in a prespecified path

#     example:
#         base_path = "C:/Users/DAWS2_code/UP"
#         method = "vertex"
#         y = create_folder(base_path, method)
#     """
#     base_path = Path(base_path)

#     res_path = base_path / method
#     res_path.mkdir(parents=True, exist_ok=True)

#     return res_path


# def create_csv(res_path, filename, data):
#     """Creates a .csv file and sotres it in a pre-specified folder with results generated by a UP method

#     args:
#         - res_path: A folder in a prespecified path named after the called UP method
#         - filename: the name of the file
#         - data: a pandas.dataframe with results from UP method

#     signature:
#         create_csv(res_path = path, filename = filename, data = pandas.dataframe) -> path.filename

#     note:
#         - the augument `res_path` will specify the folder where the .csv file will be created.
#         - argument `file` will provide the name of hte .csv file.
#         - argument `data` will provide data in terms of pandas.dataframe.

#     return:
#         -  A .csv file in a prespecified folder

#     example:
#         base_path = "C:/Users/DAWS2_code/UP/vertex"
#         filename = 'min_max_values'
#        df = pd.DataFrame(
#          {"Name" : ["y0", "y0"],
#           "fun"  : ["min","max"]
#           "y0" : [4, 6]}, index = [1, 2, 3])
#         header = ['Name', 'fun', 'values']
#         y = create_csv(res_path, filename, df)
#     """
#     try:
#         # Attempt to open the file
#         file_path = res_path / filename
#         with open(file_path, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(data.columns)
#             writer.writerows(data.values.tolist())
#     except FileNotFoundError:
#         print("The file does not exist.")

#     return filename


# def save_results(data, method, res_path, fun=None):
#     if fun is None:
#         Results = post_processing(
#             data["x"], all_output=None, method=method, res_path=res_path
#         )
#     else:
#         Results = post_processing(data["x"], data.get("f"), method, res_path)

#     return Results


def condense_bounds(bounds, N):
    """
    Condenses lower and upper bounds of a probability distribution to a specified size.

    Args:
      bounds: A NumPy array of shape (num_outputs, 2, num_points) representing the lower
              and upper bounds of a probability distribution for potentially multiple outputs.
              The first dimension corresponds to different outputs of the function,
              the second dimension corresponds to lower and upper bounds (0 for lower, 1 for upper),
              and the third dimension corresponds to the original discretization points.
      N: The desired size of the condensed arrays.

    Returns:
      A NumPy array of shape (num_outputs, 2, N) containing the condensed lower and upper bounds.
    """
    num_outputs = bounds.shape[0]
    num_points = bounds.shape[2]

    # Handle different condensation sizes for each output
    if isinstance(N, int):
        N = [N] * num_outputs  # Create a list with the same size for all outputs

    # Initialize with the maximum size
    condensed_bounds = np.zeros((num_outputs, 2, max(N)))

    for i in range(num_outputs):
        interval_size = num_points // N[i]

        lower_bounds_sorted = np.sort(bounds[i, 0, :])
        upper_bounds_sorted = np.sort(bounds[i, 1, :])

        condensed_lower = np.array(
            [
                lower_bounds_sorted[j * interval_size + interval_size - 1]
                for j in range(N[i])
            ]
        )
        condensed_upper = np.array(
            [upper_bounds_sorted[j * interval_size] for j in range(N[i])]
        )

        # Assign to the correct slice
        condensed_bounds[i, 0, : N[i]] = condensed_lower
        condensed_bounds[i, 1, : N[i]] = condensed_upper

    return condensed_bounds
