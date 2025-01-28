import numpy as np
from typing import Callable, Union
import tqdm
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import cartesian
from pyuncertainnumber.propagation.epistemic_uncertainty.extreme_point_func import extreme_pointX
from pyuncertainnumber.propagation.epistemic_uncertainty.extremepoints import extremepoints_method
from pyuncertainnumber.propagation.utils import Propagation_results, condense_bounds

# TODO add tail concentrating algorithms.
# TODO add x valus for min and max


def second_order_propagation_method(x: list, f: Callable = None,
                                    results: Propagation_results = None,
                                    method='endpoints',
                                    n_disc: Union[int, np.ndarray] = 10,
                                    condensation: Union[float,
                                                        np.ndarray] = None,
                                    tOp: Union[float, np.ndarray] = 0.999,
                                    bOt: Union[float, np.ndarray] = 0.001,
                                    save_raw_data='no') -> Propagation_results:  # Specify return type
    """
    args:
        - x (list): A list of `UncertainNumber` objects representing the uncertain inputs.
        - f (Callable): The function to evaluate.
        - results (Propagation_results, optional): An object to store propagation results.
                                    Defaults to None, in which case a new
                                    `Propagation_results` object is created.
        - method (str, optional): The method used to estimate the bounds of each combination 
                            of focal elements. Can be either 'endpoints' or 'extremepoints'. 
                            Defaults to 'endpoints'.
        - n_disc (Union[int, np.ndarray], optional): The number of discretization points 
                                    for each uncertain input. If an integer is provided,
                                    it is used for all inputs. If a NumPy array is provided,
                                    each element specifies the number of discretization 
                                    points for the corresponding input. 
                                    Defaults to 10.
        - condensation (Union[float, np.ndarray], optional): A parameter or array of parameters 
                                    to control the condensation of the output p-boxes. 
                                    Defaults to None.
        - tOp (Union[float, np.ndarray], optional): Upper threshold or array of thresholds for 
                                    discretization. 
                                    Defaults to 0.999.
        - bOt (Union[float, np.ndarray], optional): Lower threshold or array of thresholds for 
                                    discretization. 
                                    Defaults to 0.001.
        - save_raw_data (str, optional): Whether to save raw data ('yes' or 'no'). 
                                   Defaults to 'no'.

    signature:
        second_order_propagation_method(x: list, f: Callable, results: Propagation_results = None, ...) -> Propagation_results

    notes:
        - Performs second-order uncertainty propagation for mixed uncertain numbers.
        - The function handles different types of uncertain numbers (distributions, intervals, 
          p-boxes) and discretizes them accordingly.
        - It generates combinations of focal elements from the discretized uncertain inputs.
        - For the 'endpoints' method, it evaluates the function at all combinations of endpoints 
          of the focal elements.
        - For the 'extremepoints' method, it uses the `extremepoints_method` to determine the 
          signs of the partial derivatives and evaluates the function at the extreme points.
        - The output p-boxes are constructed by considering the minimum and maximum values obtained 
          from the function evaluations.
        - The `condensation` parameter can be used to reduce the number of intervals in the output 
          p-boxes. 

    returns:
        Propagation_results:  A `Propagation_results` object containing:
            - raw_data (dict): Dictionary containing raw data shared across output(s):
                    - x (np.ndarray): Input values.
                    - f (np.ndarray): Output values.
                    - min (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the minimum of that output.
                    - max (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the maximum of that output.
                    - bounds (np.ndarray): 2D array of lower and upper bounds for each output.

    example:
        from pyuncertainnumber import UncertainNumber

        def Fun(x):

            input1= x[0]
            input2=x[1]
            input3=x[2]
            input4=x[3]
            input5=x[4]

            output1 = input1 + input2 + input3 + input4 + input5
            output2 = input1 * input2 * input3 * input4 * input5

            return np.array([output1, output2])

        means = np.array([1, 2, 3, 4, 5])
        stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        x = [
            UncertainNumber(essence = 'distribution', distribution_parameters= ["gaussian",[means[0], stds[0]]]),

            UncertainNumber(essence = 'interval', bounds= [means[1]-2* stds[1], means[1]+2* stds[1]]),
            UncertainNumber(essence = 'interval', bounds= [means[2]-2* stds[2], means[2]+2* stds[2]]),
            UncertainNumber(essence = 'interval', bounds= [means[3]-2* stds[3], means[3]+2* stds[3]]),
            UncertainNumber(essence = 'interval', bounds= [means[4]-2* stds[4], means[4]+2* stds[4]])
            ]

        results = second_order_propagation_method(x=x, f=Fun, method = 'endpoints', n_disc= 5)

    """
    d = len(x)  # dimension of uncertain numbers
    results = Propagation_results()
    bounds_x = []
    ranges = np.zeros((2, d))
    u_list = []  # List to store 'u' for each uncertain number

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence != "interval":
            # perform outward directed discretisation
            if isinstance(n_disc, int):
                nd = n_disc + 1
            else:
                nd = n_disc[i] + 1  # Use the i-th element of n_disc array

            distribution_family = un.distribution_parameters[0]

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:
                if isinstance(tOp, np.ndarray):
                    top = tOp[i]
                else:
                    top = tOp
                if isinstance(bOt, np.ndarray):
                    bot = bOt[i]
                else:
                    bot = bOt
                u = np.linspace(bot, top, nd)  # Use bOt and tOp here

        else:  # un.essence == "interval"
            u = np.array([0.0, 1.0])  # Or adjust as needed for intervals

        u_list.append(u)  # Add 'u' to the list

        # Generate discrete p-boxes
        match un.essence:
            case "distribution":
                # Calculate xl and xr for distributions (adjust as needed)
                # Assuming un.ppf(u) returns a list or array
                temp_xl = un.ppf(u_list[i][:-1])
                # Adjust based on your distribution
                temp_xr = un.ppf(u_list[i][1:])
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case "interval":
                # Repeat lower bound for all quantiles
                temp_xl = np.array([un.bounds[0]])
                # Repeat upper bound for all quantiles
                temp_xr = np.array([un.bounds[1]])
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)
                ranges[:, i] = un.bounds  # np.array([un.bounds])

            case "pbox":
                temp_xl = un.ppf(u_list[i][:-1])[0]
                temp_xr = un.ppf(u_list[i][1:])[1]
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

    # Automatically generate merged_array_index
    bounds_x_index = [np.arange(len(sub_array)) for sub_array in bounds_x]

    # Calculate Cartesian product of indices using your cartesian function
    cartesian_product_indices = cartesian(*bounds_x_index)

    # Generate the final array using the indices
    focal_elements_comb = []
    for indices in cartesian_product_indices:
        temp = []
        for i, index in enumerate(indices):
            temp.append(bounds_x[i][index])
        focal_elements_comb.append(temp)

    focal_elements_comb = np.array(focal_elements_comb, dtype=object)
    all_output = None

    if f is not None:
        # Efficiency upgrade: store repeated evaluations
        inpsList = np.zeros((0, d))
        evalsList = []
        numRun = 0

        match method:
            case "endpoints" | "second_order_endpoints":
                # Pre-allocate the array
                x_combinations = np.empty(
                    (focal_elements_comb.shape[0]*(2**d), d), dtype=float)
                current_index = 0  # Keep track of the current insertion index

                for array in focal_elements_comb:
                    cartesian_product_x = cartesian(*array)
                    # Get the number of combinations from cartesian(*array)
                    num_combinations = cartesian_product_x.shape[0]

                    # Assign the cartesian product to the appropriate slice of x_combinations
                    x_combinations[current_index: current_index +
                                   num_combinations] = cartesian_product_x
                    current_index += num_combinations  # Update the insertion index

                # Initialize all_output as a list to store outputs initially
                all_output_list = []
                evalsList = []
                numRun = 0
                # Initialize inpsList with the correct number of columns
                inpsList = np.empty((0, x_combinations.shape[1]))

                # Wrap the loop with tqdmx_combinations
                for case in tqdm.tqdm(x_combinations, desc="Evaluating focal points"):
                    im = np.where((inpsList == case).all(axis=1))[0]
                    if not im.size:
                        output = f(case)
                        all_output_list.append(output)
                        inpsList = np.vstack([inpsList, case])
                        evalsList.append(output)
                        numRun += 1
                    else:
                        all_output_list.append(evalsList[im[0]])

                # Determine num_outputs AFTER running the function
                try:
                    num_outputs = len(all_output_list[0])
                except TypeError:
                    num_outputs = 1

                # Convert all_output to a 2D NumPy array
                all_output = np.array(all_output_list).reshape(
                    focal_elements_comb.shape[0], (2**d), num_outputs)

                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=1)
                max_values = np.max(all_output, axis=1)

                lower_bound = np.zeros((num_outputs, len(min_values)))
                upper_bound = np.zeros((num_outputs, len(max_values)))

                bounds = np.empty((num_outputs, 2, lower_bound.shape[1]))

                for i in range(num_outputs):
                    min_values[:, i] = np.sort(min_values[:, i])
                    max_values[:, i] = np.sort(max_values[:, i])
                    lower_bound[i, :] = min_values[:, i]  # Extract each column
                    upper_bound[i, :] = max_values[:, i]

                    bounds[i, 0, :] = lower_bound[i, :]
                    bounds[i, 1, :] = upper_bound[i, :]

            case "extremepoints" | "second_order_extremepoints":
                # Determine the positive or negative signs for each input

                res = extremepoints_method(ranges.T, f)

                # Determine the number of outputs from the first evaluation
                try:
                    num_outputs = res.raw_data['sign_x'].shape[0]
                except TypeError:
                    num_outputs = 1  # If f returns a single value

                inpsList = np.zeros((0, d))
                evalsList = np.zeros((0, num_outputs))
                all_output_list = []

                # Preallocate all_output_list with explicit loops
                all_output_list = []
                for _ in range(num_outputs):
                    output_for_current_output = []
                    # Changed to focal_elements_comb
                    for _ in range(len(focal_elements_comb)):
                        output_for_current_output.append([None, None])
                    all_output_list.append(output_for_current_output)

                # Iterate over focal_elements_comb
                for i, slice in tqdm.tqdm(enumerate(focal_elements_comb), desc="Evaluating focal points", total=len(focal_elements_comb)):
                    # Changed to 2 for the two extreme points
                    Xsings = np.empty((2, d))
                    # Use the entire sign_x array
                    Xsings[:, :] = extreme_pointX(
                        slice, res.raw_data['sign_x'])

                    for k in range(Xsings.shape[0]):  #
                        c = Xsings[k, :]
                        im = np.where((inpsList == c).all(axis=1))[0]
                        if not im.size:
                            output = f(c)

                            # Store each output in a separate sublist
                            for out in range(num_outputs):
                                # Changed indexing
                                all_output_list[out][i][k] = output[out]

                            inpsList = np.vstack([inpsList, c])

                            # Ensure output is always a NumPy array
                            if not isinstance(output, np.ndarray):
                                output = np.array(output)

                            evalsList = np.vstack([evalsList, output])
                        else:
                            for out in range(num_outputs):
                                # Changed indexing
                                all_output_list[out][i][k] = evalsList[im[0]][out]

                # Reshape all_output based on the actual number of elements per output
                all_output = np.array(all_output_list, dtype=object)
                all_output = np.reshape(
                    all_output, (num_outputs, -1, 2))  # Reshape to 3D

                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=2)
                max_values = np.max(all_output, axis=2)

                lower_bound = np.zeros((num_outputs, min_values.shape[1]))
                upper_bound = np.zeros((num_outputs, max_values.shape[1]))

                bounds = np.empty((num_outputs, 2, lower_bound.shape[1]))

                for i in range(num_outputs):
                    lower_bound[i, :] = np.sort(
                        min_values[i, :])  # Extract each column
                    upper_bound[i, :] = np.sort(max_values[i, :])

                    bounds[i, 0, :] = lower_bound[i, :]
                    bounds[i, 1, :] = upper_bound[i, :]

            case _:
                raise ValueError(
                    "Invalid UP method! endpoints_cauchy are under development.")

        if condensation is not None:
            bounds = condense_bounds(bounds, condensation)

        results.raw_data['bounds'] = bounds
        results.raw_data['min'] = np.array([{'f': lower_bound[i, :]} for i in range(
            num_outputs)])  # Initialize as a NumPy array
        results.raw_data['max'] = np.array([{'f': upper_bound[i, :]} for i in range(
            num_outputs)])  # Initialize as a NumPy array

        if save_raw_data == 'yes':
            print('No raw data provided for this method!')
            # results.add_raw_data(f= all_output, x= x_combinations)

    elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(f=None, x=x_combinations)

    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results
