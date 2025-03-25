import numpy as np
from typing import Callable, Union
import tqdm
from pyuncertainnumber.propagation.epistemic_uncertainty.extreme_point_func import extreme_pointX
from pyuncertainnumber.propagation.epistemic_uncertainty.extremepoints import extremepoints_method
from pyuncertainnumber.propagation.epistemic_uncertainty.local_optimisation import local_optimisation_method
from pyuncertainnumber.propagation.epistemic_uncertainty.genetic_optimisation import genetic_optimisation_method
from pyuncertainnumber.propagation.utils import Propagation_results, condense_bounds

def imp(X):
    """Imposition of intervals."""

    return np.array([np.max(X[:, 0]), np.min(X[:, 1])])

def varied_discretisation_propagation_method(
    x: list,
    f: Callable = None,
    results: Propagation_results = None,
    method: str = 'extremepoints',
    n_disc: Union[int, np.ndarray] = 10,
    x0: Union[float, np.ndarray] = None,                              
    tol_loc: Union[float, np.ndarray] = None, options_loc: dict = None,
    method_loc: str ='Nelder-Mead',
    pop_size: Union[int, np.ndarray] = 1500,
    n_gen: Union[int, np.ndarray]= 150,
    tol: Union[int, np.ndarray] = 1e-3,
    n_gen_last: Union[int, np.ndarray] = 10,
    tOp: Union[float, np.ndarray] = 0.999,
    bOt: Union[float, np.ndarray] = 0.001,
    condensation: Union[float, np.ndarray] = None,
) -> Propagation_results:
    """
    Performs uncertainty propagation for mixed uncertain numbers.

    args:
        x (list): A list of `UncertainNumber` objects representing the uncertain inputs.
        f (Callable): The function to evaluate.
        results (Propagation_results, optional): An object to store propagation results.
                                                Defaults to None, in which case a new
                                                `Propagation_results` object is created.
        method (str, optional): Available methods:
                                - 'varied_discretisation_extremepoints':
                                - 'varied_discretisation_local_opt' or 'varied_discretisation_local_optimisation': Uses local optimization
                                  (e.g., Nelder-Mead) to find the minimum and maximum values within each interval combination.
                                - 'varied_discretisation_genetic_opt' or 'varied_discretisation_genetic_optimisation': Uses a genetic algorithm
                                    to find the minimum and maximum values within each interval combination.
                            Defaults to 'varied_discretisation_extremepoints'.
        n_disc (Union[int, np.ndarray], optional): The number of discretization points
                                                for each uncertain input. If an integer
                                                is provided, it is used for all inputs.
                                                If a NumPy array is provided, each element
                                                specifies the number of discretization
                                                points for the corresponding input.
                                                Defaults to 10.
        x0 (Union[float, np.ndarray], optional): Initial guess for local optimization methods.
        tol_loc (Union[float, np.ndarray], optional): Tolerance for convergence in local optimization.
        options_loc (dict, optional): Options for local optimization.
        method_loc (str, optional): The local optimization method to use (e.g., 'Nelder-Mead'). Defaults to 'Nelder-Mead'.
        pop_size (Union[int, np.ndarray], optional): Population size for genetic optimization. Defaults to 1500.
        n_gen (Union[int, np.ndarray], optional): Number of generations for genetic optimization. Defaults to 150.
        tol (Union[int, np.ndarray], optional): Tolerance for convergence in genetic optimization. Defaults to 1e-3.
        n_gen_last (Union[int, np.ndarray], optional): Number of generations without improvement for convergence
                                                     in genetic optimization. Defaults to 10.                                                
        tOp (Union[float, np.ndarray], optional): Upper threshold or array of thresholds
                                                for discretization.
                                                Defaults to 0.999.
        bOt (Union[float, np.ndarray], optional): Lower threshold or array of thresholds
                                                for discretization.
                                                Defaults to 0.001.
        condensation (Union[float, np.ndarray], optional): A parameter or array of
                                                parameters to control the condensation
                                                of the output p-boxes.
                                                Defaults to None.

    signature:
        varied_discretisation_propagation_method(x: list, f: Callable, ...) -> Propagation_results

    notes:
        - It is more efficient and more conservative than the
          focused_discretisation_propagation_method.
        - If 'n_disc' is an np.array, the code slices each number to the maximum number
          of 'n_disc'.
        - The function handles different types of uncertain numbers (distributions and
          p-boxes for this version) and discretizes them with the same number of
          n_disc. To ensure conservative results, the function employs an
          outward-directed discretization approach when discretizing probability
          distributions and pboxes. For distributions that extend to infinity
          (e.g., normal distribution), the discretization process incorporates
          cut-off points defined by the tOp (upper) and bOt (lower) parameters to
          bound the distribution. The function analyses the effect of each input
          individually assuming all others are reduced to intervals. It uses the
          `extremepoints` to determine the signs of the partial derivatives of the
          function. For each input, the function constructs a pbox output. It then
          combines these individual p-boxes by finding the overlapping region of
          uncertainty that is common to all of them. This overlapping region
          represents the overall uncertainty in the output(s). The `condensation`
          parameter can be used to reduce the number of intervals in the output
          p-boxes.

    returns:
        Propagation_results object(s) containing:
            - 'un': UncertainNumber object(s) to characterise the empirical pbox
              of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                - 'x' (np.ndarray): Input values.
                - 'f' (np.ndarray): Output values.
                - 'min' (np.ndarray): Array of dictionaries, one for each output,
                  containing 'f' for the minimum of that output.
                - 'max' (np.ndarray): Array of dictionaries, one for each output,
                  containing 'f' for the maximum of that output.
                - 'bounds' (np.ndarray): 2D array of lower and upper bounds for
                  each output.

    example:
        from PyUncertaiNnumber import UncertainNumber

        def Fun(x):
            input1 = x[0]
            input2 = x[1]
            input3 = x[2]
            input4 = x[3]
            input5 = x[4]

            output1 = input1 + input2 + input3 + input4 + input5
            output2 = input1 * input2 * input3 * input4 * input5

            return np.array([output1, output2])

        means = np.array([1, 2, 3, 4, 5])
        stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        x = [
            UncertainNumber(essence='distribution',
                          distribution_parameters=["gaussian", [means[0], stds[0]]]),
            UncertainNumber(essence='distribution',
                          distribution_parameters=["gaussian", [means[1], stds[1]]]),
            UncertainNumber(essence='distribution',
                          distribution_parameters=["gaussian", [means[2], stds[1]]]),
            UncertainNumber(essence='distribution',
                          distribution_parameters=["gaussian", [means[3], stds[1]]]),
            UncertainNumber(essence='distribution',
                          distribution_parameters=["gaussian", [means[4], stds[1]]])
        ]

        results = varied_discretisation_propagation_method(x=x, f=Fun, meethod = 'varied_discretisation_extremepoints', n_disc=5)
    """
    d = len(x)  # dimension of uncertain numbers
    results = Propagation_results()
    xl = []
    xr = []
    ranges = np.zeros((2, d))
    n_slices = np.zeros((d), dtype=int)
    u_list = [] # List to store 'u' for each uncertain number.

    if isinstance(n_disc, np.ndarray):
        n_disc = np.max(n_disc)

    for i, un in enumerate(x):
        print(f"Discretising variable {i + 1} with essence: {un.essence}")

        if un.essence != "interval":
            distribution_family = un.distribution_parameters

            # Perform outward-directed discretization
            nd = n_disc + 1  # Use the maximum n_disc
            n_slices[i] = nd

            if isinstance(tOp, np.ndarray):
                top = tOp[i]
            else:
                top = tOp
            if isinstance(bOt, np.ndarray):
                bot = bOt[i]
            else:
                bot = bOt

            if distribution_family == 'triang':
                u = np.linspace(0, 1, nd)
            else:
                u = np.linspace(bot, top, nd)

        else:  # un.essence == "interval"
            u = np.array()
            n_slices[i] = 2

        u_list.append(u)

        # Generate discrete p-boxes
        temp_xl = [] # Temporary list to hold xl values for the current variable
        temp_xr = [] # Temporary list to hold xr values for the current variable

        match un.essence:
            case "distribution":
                # Calculate xl and xr for distributions (adjust as needed)
                # Assuming un.ppf(u) returns a list or array
                temp_xl = un.ppf(u_list[i][:-1]).tolist()
                # Adjust based on your distribution
                temp_xr = un.ppf(u_list[i][1:]).tolist()
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])

            case "interval":
                # Repeat lower bound for all quantiles
                temp_xl = np.array([un.bounds[0]]).tolist()
                # Repeat upper bound for all quantiles
                temp_xr = np.array([un.bounds[1]]).tolist()
                ranges[:, i] = np.array([un.bounds])

            case "pbox":
                # Assuming un.ppf(u) returns a list of lists
                temp_xl = un.ppf(u_list[i][:-1])[0].tolist()
                temp_xr = un.ppf(u_list[i][1:])[1].tolist()
                ranges[:, i] = np.array([temp_xl[0], temp_xr[-1]])
            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

        xl.append(temp_xl)  # Add the temporary list to xl
        xr.append(temp_xr)  # Add the temporary list to xr

    match method:
        case "extremepoints" | "varied_discretisation_extremepoints":
            if f is not None:
                # Determine the positive or negative signs for each input
                res = extremepoints_method(ranges.T, f)

                # Determine the number of outputs from the first evaluation
                try:
                    num_outputs = res.raw_data['part_deriv_sign'].shape[0]
                except TypeError:
                    num_outputs = 1  # If f returns a single value

                all_output_list = []
                for _ in range(num_outputs):
                    output_for_current_output = []
                    for input in range(d):
                        slices_for_input = []
                        for _ in range(n_slices[input] - 1):  # Use n_slices[input] here
                            slices_for_input.append([None, None])
                        output_for_current_output.append(slices_for_input)
                    all_output_list.append(output_for_current_output)

                inpsList = np.zeros((0, d))
                evalsList = np.zeros((0, num_outputs))

                for input in range(d):  # Iterate over input variables first
                    X = [ranges[:, k].tolist() for k in range(d)]
                    temp_X = X.copy()
                    Xsings = np.empty((n_slices[input], 2, d))

                    for slice in tqdm.tqdm(range(n_slices[input] - 1),
                                        desc=f"Processing input {input + 1}"):
                        temp_X[input] = []
                        temp_X[input].extend(
                            np.array([xl[input][slice], xr[input][slice]]).tolist())
                        rang = np.array([temp_X[i] for i in range(d)], dtype=object)
                        Xsings[slice, :, :] = extreme_pointX(rang,
                                                            res.raw_data['part_deriv_sign'])  # Use the entire part_deriv_sign array

                        for k in range(Xsings.shape[1]):
                            c = Xsings[slice, k, :]
                            im = np.where((inpsList == c).all(axis=1))[0]
                            if not im.size:
                                output = f(c)
                                inpsList = np.vstack([inpsList, c])
                                evalsList = np.vstack([evalsList, np.array(output)])
                                for out in range(num_outputs):
                                    all_output_list[out][input][slice][k] = output[out]
                            else:
                                for out in range(num_outputs):
                                    all_output_list[out][input][slice][k] = evalsList[im[0]][out]

                print(f"Number of total function evaluations: {len(inpsList)}")

                # Reshape all_output based on the actual number of elements per output
                all_output = np.array(all_output_list, dtype=object)

                all_output = np.reshape(
                    all_output, (num_outputs, d, -1, 2))  # Reshape to 4D

                # Calculate min and max for each output and input variable
                min_values = np.min(all_output, axis=3)
                max_values = np.max(all_output, axis=3)

                min_val = np.sort(min_values, axis=2)
                max_val = np.sort(max_values, axis=2)

                # Initialize bounds_input
                bounds_input = np.empty((num_outputs, d, n_disc, 2))

                for i in range(num_outputs):
                    for j in range(d):  # Iterate over input variables
                        # Merge min_values and max_values into bounds_input
                        for k in range(n_disc):
                            bounds_input[i, j, k, 0] = min_val[i, j, k]
                            bounds_input[i, j, k, 1] = max_val[i, j, k]

                        # Sort bounds_input along the last axis (k)
                        bounds_input[i, j, :, :] = np.sort(bounds_input[i, j, :, :], axis=-1)

                bounds = np.empty((num_outputs, 2, n_disc))  # Initialize bounds
                lower_bound = np.zeros((num_outputs, n_disc))  # Initialize lower_bound
                upper_bound = np.zeros((num_outputs, n_disc))  # Initialize upper_bound

                for i in range(num_outputs):
                    temp_bounds = []# Temporary list for bounds

                    for k in range(n_disc):  # Iterate over input variables
                        # Impose the p-boxes for each input variable and store in temporary list
                        temp_bounds.append(imp(bounds_input[i, :, k, :]))

                    # Impose the p-boxes across all input variables for the current output
                    bounds[i, :, :] = np.array(temp_bounds).T

                    # Extract lower_bound and upper_bound from bounds
                    lower_bound[i, :] = bounds[i, 1, :]
                    upper_bound[i, :] = bounds[i, 0, :]

                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)

                results.raw_data['bounds'] = bounds
                results.raw_data['min'] = np.array([{'f': bounds[i, 0, :]} for i in
                                                   range(num_outputs)])  # Initialize as a NumPy array
                results.raw_data['max'] = np.array([{'f': bounds[i, 1, :]} for i in
                                                   range(num_outputs)])  # Initialize as a NumPy array

            else:
                raise ValueError("No function is provided. Please provide a function!")

        case "varied_discretisation_local_optimisation"|"varied_discretisation_local_opt":
            if f is not None:
                num_outputs = 1  # If f returns a single value

                all_output_list = []
                for _ in range(num_outputs):
                    output_for_current_output = []
                    for input in range(d):
                        slices_for_input = []
                        for _ in range(n_slices[input] - 1):  # Use n_slices[input] here
                            slices_for_input.append([None, None])
                        output_for_current_output.append(slices_for_input)
                    all_output_list.append(output_for_current_output)
                
                all_messages_min = [[([''] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
                all_messages_max = [[([''] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
                all_combined_messages  = [[([''] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
                all_input_list = [[([None] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
   
                for input_index in range(d):  # Iterate over input variables first
                    X = [ranges[:, k].tolist() for k in range(d)]
                    temp_X = X.copy()

                    for slice_index in tqdm.tqdm(range(n_slices[input_index] - 1),
                                        desc=f"Processing input {input_index + 1}"):
                        temp_X[input_index] = []
                        temp_X[input_index].extend(
                            np.array([xl[input_index][slice_index], xr[input_index][slice_index]]).tolist())
                        rang = np.array([temp_X[i] for i in range(d)], dtype=object) 

                        # Store the interval 'rang'
                        all_input_list[0][input_index][slice_index] = rang.copy()
                        
                        local_opt_results = local_optimisation_method(x = rang, f=f,results = None, 
                                       x0=x0,                              
                                       tol_loc = tol_loc, options_loc = options_loc, method_loc= method_loc)
                    
                        # Extract results and inputs
                        min_result = local_opt_results.raw_data['min'][0]
                        max_result = local_opt_results.raw_data['max'][0]

                        all_output_list[0][input_index][slice_index] = np.array([[min_result['f'], max_result['f']]])
                        min_msg = min_result['message']
                        max_msg = max_result['message']
                        all_messages_min[0][input_index][slice_index] = min_msg
                        all_messages_max[0][input_index][slice_index] = max_msg 

                        
                        # Create the combined message
                        if min_msg == "Optimization terminated successfully" and max_msg == "Optimization terminated successfully":
                            all_combined_messages[0][input_index][slice_index] = "Optimisation terminated successfully"
                        elif min_msg == max_msg:
                            all_combined_messages[0][input_index][slice_index] = min_msg
                        else:
                            all_combined_messages[0][input_index][slice_index] = f"Min: {min_msg}, Max: {max_msg}"

                
                # Reshape all_output based on the actual number of elements per output
                all_output = np.array(all_output_list, dtype=object)

                results.add_raw_data(x= all_input_list)
                results.add_raw_data(f= all_output)
                results.add_raw_data(message= all_combined_messages)

                all_output = np.reshape(
                    all_output, (num_outputs, d, -1, 2))  # Reshape to 4D

                # Calculate min and max for each output and input variable
                min_values = np.min(all_output, axis=3)
                max_values = np.max(all_output, axis=3)

                min_val = np.sort(min_values, axis=2)
                max_val = np.sort(max_values, axis=2)

                # Initialize bounds_input
                bounds_input = np.empty((num_outputs, d, n_disc, 2))

                for i in range(num_outputs):
                    for j in range(d):  # Iterate over input variables
                        # Merge min_values and max_values into bounds_input
                        for k in range(n_disc):
                            bounds_input[i, j, k, 0] = min_val[i, j, k]
                            bounds_input[i, j, k, 1] = max_val[i, j, k]

                        # Sort bounds_input along the last axis (k)
                        bounds_input[i, j, :, :] = np.sort(bounds_input[i, j, :, :], axis=-1)

                bounds = np.empty((num_outputs, 2, n_disc))  # Initialize bounds
                lower_bound = np.zeros((num_outputs, n_disc))  # Initialize lower_bound
                upper_bound = np.zeros((num_outputs, n_disc))  # Initialize upper_bound

                for i in range(num_outputs):
                    temp_bounds = []# Temporary list for bounds

                    for k in range(n_disc):  # Iterate over input variables
                        # Impose the p-boxes for each input variable and store in temporary list
                        temp_bounds.append(imp(bounds_input[i, :, k, :]))

                    # Impose the p-boxes across all input variables for the current output
                    bounds[i, :, :] = np.array(temp_bounds).T

                    # Extract lower_bound and upper_bound from bounds
                    lower_bound[i, :] = bounds[i, 1, :]
                    upper_bound[i, :] = bounds[i, 0, :]

                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)

                results.raw_data['bounds'] = bounds
                results.raw_data['min'] = np.array([{'f': bounds[i, 0, :]} for i in
                                                   range(num_outputs)])  # Initialize as a NumPy array
                results.raw_data['max'] = np.array([{'f': bounds[i, 1, :]} for i in
                                                   range(num_outputs)])  # Initialize as a NumPy array

            else:
                raise ValueError("No function is provided. Please provide a function!")

        case "varied_discretisation_genetic_optimisation"|"varied_discretisation_genetic_opt":
            if f is not None:
                num_outputs = 1  # If f returns a single value

                all_output_list = []
                for _ in range(num_outputs):
                    output_for_current_output = []
                    for input in range(d):
                        slices_for_input = []
                        for _ in range(n_slices[input] - 1):  # Use n_slices[input] here
                            slices_for_input.append([None, None])
                        output_for_current_output.append(slices_for_input)
                    all_output_list.append(output_for_current_output)
                
                all_messages_min = [[([''] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
                all_messages_max = [[([''] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
                all_combined_messages  = [[([''] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
                all_input_list = [[([None] * (n_slices[i] - 1)) for _ in range(d)] for _ in range(num_outputs)]
   
                for input_index in range(d):  # Iterate over input variables first
                    X = [ranges[:, k].tolist() for k in range(d)]
                    temp_X = X.copy()

                    for slice_index in tqdm.tqdm(range(n_slices[input_index] - 1),
                                        desc=f"Processing input {input_index + 1}"):
                        temp_X[input_index] = []
                        temp_X[input_index].extend(
                            np.array([xl[input_index][slice_index], xr[input_index][slice_index]]).tolist())
                        rang = np.array([temp_X[i] for i in range(d)], dtype=object) 

                        # Store the interval 'rang'
                        all_input_list[0][input_index][slice_index] = rang.copy()
                        
                        local_opt_results = genetic_optimisation_method(x_bounds = rang, f=f,results = None, 
                                        pop_size=pop_size, n_gen=n_gen, tol=tol,
                                        n_gen_last=n_gen_last
                                    )
                    
                        # Extract results and inputs
                        min_result = local_opt_results.raw_data['min'][0]
                        max_result = local_opt_results.raw_data['max'][0]

                        all_output_list[0][input_index][slice_index] = np.array([[min_result['f'], max_result['f']]])
                        min_msg = min_result['message']
                        max_msg = max_result['message']
                        all_messages_min[0][input_index][slice_index] = min_msg
                        all_messages_max[0][input_index][slice_index] = max_msg 

                        
                        # Create the combined message
                        if min_msg == "Optimization terminated successfully" and max_msg == "Optimization terminated successfully":
                            all_combined_messages[0][input_index][slice_index] = "Optimisation terminated successfully"
                        elif min_msg == max_msg:
                            all_combined_messages[0][input_index][slice_index] = min_msg
                        else:
                            all_combined_messages[0][input_index][slice_index] = f"Min: {min_msg}, Max: {max_msg}"

                
                # Reshape all_output based on the actual number of elements per output
                all_output = np.array(all_output_list, dtype=object)

                results.add_raw_data(x= all_input_list)
                results.add_raw_data(f= all_output)
                results.add_raw_data(message= all_combined_messages)

                all_output = np.reshape(
                    all_output, (num_outputs, d, -1, 2))  # Reshape to 4D

                # Calculate min and max for each output and input variable
                min_values = np.min(all_output, axis=3)
                max_values = np.max(all_output, axis=3)

                min_val = np.sort(min_values, axis=2)
                max_val = np.sort(max_values, axis=2)

                # Initialize bounds_input
                bounds_input = np.empty((num_outputs, d, n_disc, 2))

                for i in range(num_outputs):
                    for j in range(d):  # Iterate over input variables
                        # Merge min_values and max_values into bounds_input
                        for k in range(n_disc):
                            bounds_input[i, j, k, 0] = min_val[i, j, k]
                            bounds_input[i, j, k, 1] = max_val[i, j, k]

                        # Sort bounds_input along the last axis (k)
                        bounds_input[i, j, :, :] = np.sort(bounds_input[i, j, :, :], axis=-1)

                bounds = np.empty((num_outputs, 2, n_disc))  # Initialize bounds
                lower_bound = np.zeros((num_outputs, n_disc))  # Initialize lower_bound
                upper_bound = np.zeros((num_outputs, n_disc))  # Initialize upper_bound

                for i in range(num_outputs):
                    temp_bounds = []# Temporary list for bounds

                    for k in range(n_disc):  # Iterate over input variables
                        # Impose the p-boxes for each input variable and store in temporary list
                        temp_bounds.append(imp(bounds_input[i, :, k, :]))

                    # Impose the p-boxes across all input variables for the current output
                    bounds[i, :, :] = np.array(temp_bounds).T

                    # Extract lower_bound and upper_bound from bounds
                    lower_bound[i, :] = bounds[i, 1, :]
                    upper_bound[i, :] = bounds[i, 0, :]

                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)

                results.raw_data['bounds'] = bounds
                results.raw_data['min'] = np.array([{'f': bounds[i, 0, :]} for i in
                                                   range(num_outputs)])  # Initialize as a NumPy array
                results.raw_data['max'] = np.array([{'f': bounds[i, 1, :]} for i in
                                                   range(num_outputs)])  # Initialize as a NumPy array

            else:
                raise ValueError("No function is provided. Please provide a function!")
               
        case _:
            raise ValueError(
                "Invalid UP method!")

    return results

