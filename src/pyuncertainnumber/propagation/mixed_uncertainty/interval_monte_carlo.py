import numpy as np
from typing import Callable, Union
import tqdm
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import cartesian
from pyuncertainnumber.propagation.utils import Propagation_results, condense_bounds

#TODO add tail concentrating algorithms.
#TODO add x valus for min and max
def interval_monte_carlo_method(x: list, f:Callable = None,  
                                    results: Propagation_results = None, 
                                    method = 'endpoints',
                                    n_sam: int = 500,
                                    condensation: Union[float, np.ndarray] = None, 
                                    save_raw_data= 'no')-> Propagation_results:  # Specify return type
    
    """ This function performs uncertainty propagation for a mix of uncertain numbers. It is designed to 
        handle situations where there are different types of uncertainty in the model's inputs, such as probability 
        distributions, intervals, and p-boxes. To ensure conservative results, the function employs an outward-directed 
        discretization approach for probability distributions and pboxes.  For distributions that extend to infinity 
        (e.g., normal distribution), the discretization process incorporates cut-off points defined by the tOp (upper) 
        and bOt (lower) parameters to bound the distribution. The function generates the cartesian product of the focal
        elements from the discretized uncertain inputs.
            - For the 'endpoints' method, it evaluates the function at all combinations of endpoints 
              of the focal elements.
            - For the 'extremepoints' method, it uses the `extremepoints_method` to determine the 
              signs of the partial derivatives and evaluates the function at the extreme points.
        The output p-boxes are constructed by considering the minimum and maximum values obtained from the function evaluations
        `condensation` can be used to reduce the number of intervals in the output p-boxes. 

    args:
        x (list): A list of `UncertainNumber` objects representing the uncertain inputs.
        f (Callable): The function to evaluate.
        results (Propagation_results, optional): An object to store propagation results.
                                    Defaults to None, in which case a new
                                    `Propagation_results` object is created.
        method (str, optional): The method used to estimate the bounds of each combination 
                            of focal elements. Can be either 'endpoints' or 'extremepoints'. 
                            Defaults to 'endpoints'.
        n_disc (Union[int, np.ndarray], optional): The number of discretization points 
                                    for each uncertain input. If an integer is provided,
                                    it is used for all inputs. If a NumPy array is provided,
                                    each element specifies the number of discretization 
                                    points for the corresponding input. 
                                    Defaults to 10.
        tOp (Union[float, np.ndarray], optional): Upper threshold or array of thresholds for 
                                    discretization. 
                                    Defaults to 0.999.
        bOt (Union[float, np.ndarray], optional): Lower threshold or array of thresholds for 
                                    discretization. 
                                    Defaults to 0.001.
        condensation (Union[float, np.ndarray], optional): A parameter or array of parameters 
                                    to control the condensation of the output p-boxes. 
                                    Defaults to None.
        save_raw_data (str, optional): Whether to save raw data ('yes' or 'no'). 
                                   Defaults to 'no'.

    signature:
        focused_discretisation_propagation_method(x: list, f: Callable, results: propagation_results = None, ...) -> propagation_results      
    
    raises:
      ValueError for unsupported uncertainty type, invalid UP method and if no fnction is given and saw_raw_data = 'no'
       is selected.
    
    returns:
        Returns `Propagation_results` object(s) containing:
            - 'un': UncertainNumber object(s) to characterise the empirical pbox(es) of the output(s).
            - 'raw_data' (dict): Dictionary containing raw data shared across output(s):
                    - 'x' (np.ndarray): Input values.
                    - 'f' (np.ndarray): Output values.
                    - 'min' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the minimum of that output.
                    - 'max' (np.ndarray): Array of dictionaries, one for each output,
                              containing 'f' for the maximum of that output.
                    - 'bounds' (np.ndarray): 2D array of lower and upper bounds for each output.

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
    
        results = interval_monte_carlo_method(x=x, f=Fun, method = 'endpoints', n_disc= 5)
    
    """
    num_uncertainties = len(x)
    bounds_x = np.zeros((num_uncertainties, n_sam, 2))  # Preallocate the NumPy array
    u_list = []

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence != "interval":
            u = np.random.uniform(size=n_sam)
        else:
            u = np.linspace(0, 1, n_sam)

        u_list.append(u)

        match un.essence:
            case "distribution":
                temp = un.ppf(u_list[i][:])
                bounds_x[i, :, 0] = temp
                bounds_x[i, :, 1] = temp

            case "interval":
                bounds_x[i, :, 0] = np.full(n_sam, un.bounds[0])
                bounds_x[i, :, 1] = np.full(n_sam, un.bounds[1])

            case "pbox":
                 bounds_x[i,:,0] = un.ppf(u_list[i])[:,0]
                 bounds_x[i,:,1] = un.ppf(u_list[i])[:,1]

            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

    match method:
        case "interval_mc_endpoints" | "interval_monte_carlo_endpoints":
            all_output = np.empty((n_sam, 2**num_uncertainties)) if f is not None else None
            x_combinations = np.empty((n_sam, 2**num_uncertainties, num_uncertainties))

            inpsList = np.zeros((0, num_uncertainties))
            evalsList = []
            numRun = 0

            for sample_index in tqdm.tqdm(range(n_sam), desc="Processing samples"):
                sample_intervals = [bounds_x[i, sample_index, :] for i in range(num_uncertainties)]
                cartesian_product_x = cartesian(*sample_intervals)
                x_combinations[sample_index, :, :] = cartesian_product_x

                if f is not None:
                    outputs = []
                    for case in cartesian_product_x:
                        im = np.where((inpsList == case).all(axis=1))[0]
                        if not im.size:
                            output = f(case)
                            outputs.append(output)
                            inpsList = np.vstack([inpsList, case])
                            evalsList.append(output)
                            numRun += 1
                        else:
                            outputs.append(evalsList[im[0]])
                    all_output[sample_index, :] = np.array(outputs)

            if f is not None:
                num_outputs = 1
                if isinstance(all_output[0, 0], (list, np.ndarray)):
                    num_outputs = len(all_output[0, 0])

                min_values = np.empty((n_sam, num_outputs))
                max_values = np.empty((n_sam, num_outputs))

                for i in range(n_sam):
                    if num_outputs == 1:
                        min_values[i, 0] = np.min(all_output[i, :])
                        max_values[i, 0] = np.max(all_output[i, :])
                    else:
                        for j in range(num_outputs):
                            min_values[i, j] = np.min(all_output[i, :, j])
                            max_values[i, j] = np.max(all_output[i, :, j])

                bounds = np.empty((num_outputs, 2, n_sam))
                for i in range(num_outputs):
                    bounds[i, 0, :] = min_values[:, i]
                    bounds[i, 1, :] = max_values[:, i]
            
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

                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)

                results.raw_data['bounds'] = bounds
                results.raw_data['min'] = {'f': min_values}
                results.raw_data['max'] = {'f': max_values}

                if save_raw_data == 'yes':
                    results.add_raw_data(f=all_output, x=x_combinations)

            elif save_raw_data == 'yes':
                results.add_raw_data(f=None, x=x_combinations)
            else:
                raise ValueError("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

        # case "extremepoints"| "focused_discretisation_extremepoints":
        #     # Determine the positive or negative signs for each input
        #     if f is not None:
        #         res = extremepoints_method(ranges.T, f)

        #         # Determine the number of outputs from the first evaluation
        #         try:
        #             num_outputs = res.raw_data['part_deriv_sign'].shape[0]
        #         except TypeError:
        #             num_outputs = 1  # If f returns a single value

        #         inpsList = np.zeros((0, d))
        #         evalsList = np.zeros((0, num_outputs))
        #         all_output = np.empty((num_outputs, len(intervals_comb), 2))

        #         # Preallocate all_output_list with explicit loops
        #         for i, slice in tqdm.tqdm(enumerate(intervals_comb), desc="input combinations", total=len(intervals_comb)):
        #             for out in range(num_outputs):  # Iterate over each output
        #                 for k in range(2):  # For each of the two extreme points
        #                     # Calculate Xsings using the correct part_deriv_sign for the current output
        #                     Xsings = np.empty((2, d))
        #                     Xsings[:,:] = extreme_pointX(slice, res.raw_data['part_deriv_sign'][out,:]) 

        #                     c = Xsings[k,:]
        #                     im = np.where((inpsList == c).all(axis=1))[0]
                            
        #                     if not im.size:
        #                         output = f(c)
        #                         inpsList = np.vstack([inpsList, c])
        #                         evalsList = np.vstack([evalsList, output])
        #                     else:
        #                        output = evalsList[im[0]]
                            
        #                     all_output[out, i, k] = output[out] 

        #                      # Store the specific output value

        #         # Calculate min and max efficiently
        #         min_values = np.min(all_output, axis=2)
        #         max_values = np.max(all_output, axis=2)

        #         lower_bound = np.zeros((num_outputs, min_values.shape[1]))
        #         upper_bound = np.zeros((num_outputs, max_values.shape[1]))
        #         bounds = np.empty((num_outputs, 2, lower_bound.shape[1]))

        #         for i in range(num_outputs):
        #             lower_bound[i, :] = np.sort(min_values[i, :])  # Extract each column
        #             upper_bound[i, :] = np.sort(max_values[i, :])

        #             bounds[i, 0, :] = lower_bound[i, :]
        #             bounds[i, 1, :] = upper_bound[i, :]
                
        #         if condensation is not None:
        #             bounds = condense_bounds(bounds, condensation)
                
        #         results.raw_data['bounds'] = bounds
        #         results.raw_data['min'] = {'f': np.array(lower_bound[:num_outputs,:])}  # Store min as a 2D NumPy array
        #         results.raw_data['max'] = {'f': np.array(upper_bound[:num_outputs,:])}  # Store max as a 2D NumPy array
                
        #         if save_raw_data == 'yes':
        #             #print('No raw data provided for this method!')
        #             results.add_raw_data(f= all_output, x= x_combinations)

        #     elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
        #         results.add_raw_data(f=None, x=x_combinations)

        #     else:
        #         raise ValueError(
        #             "No function is provided. Select save_raw_data = 'yes' to save the input combinations")

        case _: raise ValueError(
                     "Invalid UP method! focused_discretisation_cauchy is under development.")

    return results

from pyuncertainnumber.characterisation.uncertainNumber import UncertainNumber

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

results = interval_monte_carlo_method(x=x, f=Fun, method = 'endpoints', n_disc= 5)


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

results = interval_monte_carlo_method(x=x, f=Fun, method = 'endpoints', n_disc= 5)