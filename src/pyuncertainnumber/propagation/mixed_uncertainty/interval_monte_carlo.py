import numpy as np
from typing import Callable, Union
import tqdm
import matplotlib.pyplot as plt
from pyuncertainnumber.propagation.epistemic_uncertainty.cartesian_product import cartesian
from pyuncertainnumber.propagation.epistemic_uncertainty.extreme_point_func import extreme_pointX
from pyuncertainnumber.propagation.epistemic_uncertainty.extremepoints import extremepoints_method
from pyuncertainnumber.propagation.epistemic_uncertainty.local_optimisation import local_optimisation_method
#from pyuncertainnumber.propagation.epistemic_uncertainty.genetic_optimisation import genetic_optimisation_method
from pyuncertainnumber.propagation.utils import Propagation_results, condense_bounds

import numpy as np
from typing import Callable
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.core.callback import Callback
from pyuncertainnumber.propagation.utils import Propagation_results

def genetic_optimisation_method(x_bounds:  np.ndarray,  # Updated to list of lists
                                f: Callable = None,
                                results: Propagation_results = None,
                                pop_size: Union[int, np.ndarray] = 1500,
                                n_gen: Union[int, np.ndarray]= 150,
                                tol: Union[int, np.ndarray] = 1e-3,
                                n_gen_last: Union[int, np.ndarray] = 10
                                ) -> Propagation_results:
    """
    Performs both minimisation and maximisation using a genetic algorithm.

    args:
        x_bounds (np.ndarray): Bounds for decision variables.
        f (Callable): Objective function to optimize.
        results (Propagation_results, optional): Existing results object. Defaults to None.
        pop_size (int, list, or np.ndarray): Population size.
        n_gen (int, list, or np.ndarray): Maximum number of generations.
        tol (float, list, or np.ndarray): Tolerance for convergence check.
        n_gen_last (int, list, or np.ndarray): Number of last generations to consider for convergence.
       
    returns:
        Propagation_results: Object containing optimisation results.
    """
    class ProblemWrapper(Problem):
        """Wraps the objective function for pymoo."""
        def __init__(self, objective, **kwargs):
            super().__init__(n_obj=1, **kwargs)
            self.n_evals = 0
            self.objective = objective

        def _evaluate(self, x, out, *args, **kwargs):
            """Evaluates the objective function."""
            self.n_evals += len(x)
            out["F"] = np.array([f(xi) for xi in x]) if self.objective == 'min' else -np.array([f(xi) for xi in x])

    class ConvergenceMonitor(Callback):
        """Monitors convergence of the genetic algorithm."""
        def __init__(self, tol=1e-4, n_last=5):
            super().__init__()
            self.tol = tol
            self.n_last = n_last
            self.history = []
            self.n_generations = 0
            self.convergence_reached = False
            self.convergence_message = None

        def notify(self, algorithm):
            """Checks for convergence and updates history."""
            self.n_generations += 1
            self.history.append(algorithm.pop.get("F").min())
            if len(self.history) >= self.n_last:
                last_values = self.history[-self.n_last:]
                convergence_value = np.max(last_values) - np.min(last_values)
                if convergence_value <= self.tol and not self.convergence_reached:
                    self.convergence_message = "Convergence reached!"
                    print(self.convergence_message)
                    self.convergence_reached = True
                    algorithm.termination.force_termination = True

    def run_optimisation(objective, pop_size, n_gen, tol, n_gen_last):
        """Runs the optimisation algorithm."""
        callback = ConvergenceMonitor(tol=tol, n_last=n_gen_last)
        problem = ProblemWrapper(objective=objective, n_var=x_bounds.shape[0],
                                    xl=x_bounds[:, 0], xu=x_bounds[:, 1])
        algorithm = GA(pop_size=pop_size) #force GA algorithm
        result = minimize(problem, algorithm, ('n_gen', n_gen), callback=callback)
        return result, callback.n_generations, problem.n_evals, callback.convergence_message

    def handle_arg(arg):
        """Handles arguments that can be single values or lists."""
        if isinstance(arg, str):
            return [arg, arg]
        elif isinstance(arg, np.ndarray):
            return [int(a) for a in arg]
        elif isinstance(arg, list):
            return [int(a) for a in arg]
        else:
            return [int(arg), int(arg)]

    pop_size = handle_arg(pop_size)
    n_gen = handle_arg(n_gen)
    tol = handle_arg(tol)
    n_gen_last = handle_arg(n_gen_last)

    result_min, n_gen_min, n_iter_min, message_min = run_optimisation(
        'min', list(pop_size)[0], list(n_gen)[0], list(tol)[0], list(n_gen_last)[0]
    )

    result_max, n_gen_max, n_iter_max, message_max = run_optimisation(
        'max', list(pop_size)[1], list(n_gen)[1], list(tol)[1], list(n_gen_last)[1]
    )

    if results is None:
        results = Propagation_results()

    if not hasattr(results, 'raw_data') or results.raw_data is None:
        results.raw_data = {'min': [], 'max': []}

    results.raw_data['min']= np.append(results.raw_data['min'], {
        'x': result_min.X,
        'f': result_min.F[0],
        'message': message_min,
        'ngenerations': n_gen_min,
        'niterations': n_iter_min
    })
    
    results.raw_data['max'] = np.append(results.raw_data['max'], {
        'x': result_max.X,
        'f':  - result_max.F[0],
        'message': message_max,
        'ngenerations': n_gen_max,
        'niterations': n_iter_max
    })

    results.raw_data['bounds'] = np.array([results.raw_data['min'][0]['f'],results.raw_data['max'][0]['f']])

    return results

def interval_monte_carlo_method(x: list, f:Callable = None,  
                                    results: Propagation_results = None, 
                                    method:str = 'interval_mc_endpoints',
                                    n_sam: int = 500,
                                    x0: np.ndarray = None,                              
                                    tol_loc: np.ndarray = None, options_loc: dict = None,
                                    *, method_loc='Nelder-Mead',
                                    pop_size= np.array([1500, 1500]), n_gen=150, tol=1e-3,
                                    n_gen_last=np.array([10, 20]),
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
    
        results = focused_discretisation_propagation_method(x=x, f=Fun, method = 'endpoints', n_disc= 5)
    
    """
    d = len(x)  # dimension of uncertain numbers
    results = Propagation_results()
    bounds_x = []
    ranges = np.zeros((2, d))
    u_list = []  # List to store 'u' for each uncertain number
    convergence_issues = []# Initialize convergence_issues

    for i, un in enumerate(x):
        print(f"Processing variable {i + 1} with essence: {un.essence}")

        if un.essence != "interval":
            u = np.random.uniform(size = n_sam)

        else:  # un.essence == "interval"
            u = np.array([0.0, 1.0])  # Or adjust as needed for intervals

        u_list.append(u)  # Add 'u' to the list

        # Generate discrete p-boxes
        match un.essence:
            case "distribution":
                # Calculate xl and xr for distributions (adjust as needed)
                # Assuming un.ppf(u) returns a list or array
                temp_xl = un.ppf(u_list[i][:])
                # Adjust based on your distribution
                temp_xr = un.ppf(u_list[i][:])
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)

            case "interval":
                # Repeat lower bound for all quantiles
                temp_xl = np.array([un.bounds[0]])
                # Repeat upper bound for all quantiles
                temp_xr = np.array([un.bounds[1]])
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)

            case "pbox":
                temp_xl = un.ppf(u_list[i][:])[0]
                temp_xr = un.ppf(u_list[i][:])[1]
                # Create a 2D array of bounds
                rang = np.array([temp_xl, temp_xr]).T
                bounds_x.append(rang)

            case _:
                raise ValueError(f"Unsupported uncertainty type: {un.essence}")

    bounds_x_index = [np.arange(len(sub_array)) for sub_array in bounds_x]

    # Determine the length of the arrays that have n_sam samples
    n_sam_length = 0
    for index_array in bounds_x_index:
        if len(index_array) > 1:
            n_sam_length = len(index_array)
            break  # Assume all n_sam arrays have the same length

    # Generate the combinations of indices
    intervals_comb = []
    if n_sam_length > 0:
        for i in range(n_sam_length):
            temp = []
            for j, index_array in enumerate(bounds_x_index):
                if len(index_array) > 1:
                    temp.append(bounds_x[j][index_array[i]])
                else:
                    temp.append(bounds_x[j][0])  # Use the first (and only) element for intervals
            intervals_comb.append(temp)
    else :
        temp = []
        for j, index_array in enumerate(bounds_x_index):
            temp.append(bounds_x[j][0])
        intervals_comb.append(temp)

    intervals_comb = np.array(intervals_comb, dtype=object)
    all_output = None
   
    # Efficiency upgrade: store repeated evaluations
    inpsList = np.zeros((0, d))
    evalsList = []

    match method:
        case "interval_mc_endpoints" | "interval_monte_carlo_endpoints" :
            x_combinations = np.empty(( intervals_comb.shape[0]*(2**d), d), dtype=float)  # Pre-allocate the array
            current_index = 0  # Keep track of the current insertion index

            for array in intervals_comb:
                cartesian_product_x = cartesian(*array)
                # Get the number of combinations from cartesian(*array)
                num_combinations = cartesian_product_x.shape[0]

                # Assign the cartesian product to the appropriate slice of x_combinations
                x_combinations[current_index: current_index + num_combinations] = cartesian_product_x
                current_index += num_combinations  # Update the insertion index
            
            if f is not None:
                # Initialize all_output as a list to store outputs initially
                all_output_list = []
                evalsList = []
                num_evaluations = 0
                # Initialize inpsList with the correct number of columns
                inpsList = np.empty((0, x_combinations.shape[1]))

                # Wrap the loop with tqdmx_combinations
                for case in tqdm.tqdm(x_combinations, desc="input combinations"):
                    im = np.where((inpsList == case).all(axis=1))[0]
                    if not im.size:
                        output = f(case)
                        all_output_list.append(output)
                        inpsList = np.vstack([inpsList, case])
                        evalsList.append(output)
                        num_evaluations += 1
                    else:
                        all_output_list.append(evalsList[im[0]])

                print(f'Total number of function evaluations is: {num_evaluations}')

                # Determine num_outputs AFTER running the function
                try:
                    num_outputs = len(all_output_list[0])
                except TypeError:
                    num_outputs = 1

                # Convert all_output to a 2D NumPy array
                all_output = np.array(all_output_list).reshape(
                    intervals_comb.shape[0], (2**d), num_outputs)

                # Calculate min and max for each sublist in all_output
                min_values = np.min(all_output, axis=1)
                max_values = np.max(all_output, axis=1)

                lower_bound = np.zeros((num_outputs, len(min_values)))
                upper_bound = np.zeros((num_outputs, len(max_values)))

                bounds = np.empty((num_outputs, 2, lower_bound.shape[1]))

                for i in range(num_outputs):
                    bounds[i, 0, :] = min_values[:, i]
                    bounds[i, 1, :] = max_values[:, i]

                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)

                results.raw_data['bounds'] = bounds
                results.raw_data['min'] = np.array([{'f': bounds[i,0,:]} for i in range(
                    num_outputs)])  # Initialize as a NumPy array
                results.raw_data['max'] = np.array([{'f': bounds[i,1,:]} for i in range(
                    num_outputs)])  # Initialize as a NumPy array
                
                if save_raw_data == 'yes':
                    results.add_raw_data(f= all_output, x= x_combinations)

            elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
                results.add_raw_data(f=None, x=x_combinations)

            else:
                raise ValueError(
                    "No function is provided. Select save_raw_data = 'yes' to save the input combinations")

        case "interval_mc_extremepoints"| "interval_monte_carlo_extremepoints":
            # Determine the positive or negative signs for each input
            if f is not None:
                res = extremepoints_method(ranges.T, f)
                num_evaluations = 0

                # Determine the number of outputs from the first evaluation
                try:
                    num_outputs = res.raw_data['part_deriv_sign'].shape[0]
                except TypeError:
                    num_outputs = 1  # If f returns a single value
                
                inpsList = np.zeros((0, d))
                evalsList = np.zeros((0, num_outputs))
                all_output = np.empty((num_outputs, len(intervals_comb), 2))

                # Preallocate all_output_list with explicit loops
                for i, slice in tqdm.tqdm(enumerate(intervals_comb), desc="input combinations", total=len(intervals_comb)):
                    for out in range(num_outputs):  # Iterate over each output
                        for k in range(2):  # For each of the two extreme points
                            # Calculate Xsings using the correct part_deriv_sign for the current output
                            Xsings = np.empty((2, d))
                            Xsings[:,:] = extreme_pointX(slice, res.raw_data['part_deriv_sign'][out,:]) 

                            c = Xsings[k,:]
                            im = np.where((inpsList == c).all(axis=1))[0]
                            
                            if not im.size:
                                output = f(c)
                                inpsList = np.vstack([inpsList, c])
                                evalsList = np.vstack([evalsList, output])
                                num_evaluations += 1
                            else:
                               output = evalsList[im[0]]
                            
                            all_output[out, i, k] = output[out] 

                             # Store the specific output value

                print(f'Total number of function evaluations is: {num_evaluations + len(res)}')

                # Calculate min and max efficiently
                min_values = np.min(all_output, axis=2)
                max_values = np.max(all_output, axis=2)

                lower_bound = np.zeros((num_outputs, min_values.shape[1]))
                upper_bound = np.zeros((num_outputs, max_values.shape[1]))
                bounds = np.empty((num_outputs, 2, lower_bound.shape[1]))

                for i in range(num_outputs):
                    lower_bound[i, :] = np.sort(min_values[i, :])  # Extract each column
                    upper_bound[i, :] = np.sort(max_values[i, :])

                    bounds[i, 0, :] = lower_bound[i, :]
                    bounds[i, 1, :] = upper_bound[i, :]
                
                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)
                
                results.raw_data['bounds'] = bounds
                results.raw_data['min'] = np.array([{'f': bounds[i,0,:]} for i in range(
                    num_outputs)])  # Initialize as a NumPy array
                results.raw_data['max'] = np.array([{'f': bounds[i,1,:]} for i in range(
                    num_outputs)])  # Initialize as a NumPy array
                
                if save_raw_data == 'yes':
                    results.add_raw_data(f= all_output, x= x_combinations)

            elif save_raw_data == 'yes':  # If f is None and save_raw_data is 'yes'
                results.add_raw_data(f=None, x=x_combinations)

            else:
                raise ValueError(
                    "No function is provided. Select save_raw_data = 'yes' to save the input combinations")

        case "interval_mc_local_opt" | "interval_monte_carlo_local_optimisation" :
           
            if f is not None:
                x_min_y = None
                x_max_y= None
                message_min = None
                message_max = None

                for interval_set in tqdm.tqdm(intervals_comb, desc="Processing input combinations"):
                    inputs = np.array([np.array(interval).flatten() for interval in interval_set])
                    local_opt_results = local_optimisation_method(x = inputs, f=f,results = None, 
                              x0=x0,                              
                              tol_loc = tol_loc, options_loc = options_loc, method_loc= method_loc)
                    
                    # Extract results and inputs
                    min_result = local_opt_results.raw_data['min']
                    max_result = local_opt_results.raw_data['max']

                    if all_output is None:
                        all_output = np.array([[min_result[0]['f'], max_result[0]['f']]])
                    else:
                        all_output = np.concatenate((all_output, np.array([[min_result[0]['f'], max_result[0]['f']]])), axis=0)
                    
                    if x_min_y is None:
                        x_min_y =  np.array([min_result[0]['x']])
                    else:
                        x_min_y = np.concatenate((x_min_y, np.array([min_result[0]['x']])), axis=0) 
                    
                    if x_max_y is None:
                        x_max_y =  np.array([max_result[0]['x']])
                    else:
                        x_max_y = np.concatenate((x_max_y, np.array([max_result[0]['x']])), axis=0) 
                    
                    if message_min is None:
                        message_min =  np.array([min_result[0]['message']])
                    else:
                        message_min = np.concatenate((message_min, np.array([min_result[0]['message']])), axis=0) 
                    
                    if message_max is None:
                        message_max =  np.array([max_result[0]['message']])
                    else:
                        message_max = np.concatenate((message_max, np.array([max_result[0]['message']])), axis=0) 

        case "interval_mc_genetic_opt" | "interval_monte_carlo_genetic_optimisation" :
           
            if f is not None:
                x_min_y = None
                x_max_y= None
                message_min = None
                message_max = None

                for interval_set in tqdm.tqdm(intervals_comb, desc="Processing input combinations"):
                    inputs = np.array([np.array(interval).flatten() for interval in interval_set])
                    genetic_opt_results = genetic_optimisation_method(x_bounds = inputs, f=f,results = None, 
                                        pop_size=pop_size, n_gen=n_gen, tol=tol,
                                        n_gen_last=n_gen_last
                                    )
                    
                    # Extract results and inputs
                    min_result = genetic_opt_results.raw_data['min']
                    max_result = genetic_opt_results.raw_data['max']
                   
                    if isinstance(min_result[0]['x'], (int, float, np.float64)):
                        print("ERROR: min result x is wrong shape")
                    if isinstance(max_result[0]['x'], (int, float, np.float64)):
                        print("ERROR: max result x is wrong shape")

                    if all_output is None:
                        all_output = np.array([[min_result[0]['f'], max_result[0]['f']]])
                    else:
                        all_output = np.concatenate((all_output, np.array([[min_result[0]['f'], max_result[0]['f']]])), axis=0)
                                        
                    if x_min_y is None:
                        x_min_y =  np.array([min_result[0]['x']])
                    else:
                        x_min_y = np.concatenate((x_min_y, np.array([min_result[0]['x']])), axis=0) 
                    
                    if x_max_y is None:
                        x_max_y =  np.array([max_result[0]['x']])
                    else:
                        x_max_y = np.concatenate((x_max_y, np.array([max_result[0]['x']])), axis=0) 
                    
                    if message_min is None:
                        message_min =  np.array([min_result[0]['message']])
                    else:
                        message_min = np.concatenate((message_min, np.array([min_result[0]['message']])), axis=0) 
                    
                    if message_max is None:
                        message_max =  np.array([max_result[0]['message']])
                    else:
                        message_max = np.concatenate((message_max, np.array([max_result[0]['message']])), axis=0) 


                # Determine num_outputs AFTER running the function          
                num_outputs = 1

                # Calculate min and max for each sublist in all_output
                min_values =  all_output[:, 0]
                max_values =  all_output[:, 1]

                lower_bound = np.zeros((len(min_values)))
                upper_bound = np.zeros((len(max_values)))

                bounds = np.empty(( 2, lower_bound.shape[0]))

                index_min_values = np.argsort(min_values)
                index_max_values = np.argsort(max_values)

                x_min_y = x_min_y[index_min_values]
                x_max_y = x_max_y[index_max_values]


                lower_bound = min_values[index_min_values]  # Extract each column
                upper_bound = max_values[index_max_values]
                bounds = np.array([lower_bound, upper_bound])
                
                if condensation is not None:
                    bounds = condense_bounds(bounds, condensation)

                # Update results.raw_data with sorted x values
                results.raw_data['bounds'] = bounds

                results.raw_data['min'] = np.append(
                                            results.raw_data["min"],
                                            {"x": x_min_y, "f": lower_bound, "message": message_min},)
                results.raw_data['max'] = np.append(
                                            results.raw_data["max"],
                                            {"x": x_max_y, "f": upper_bound, "message": message_max},)

                if save_raw_data == 'yes':
                    results.add_raw_data(f=all_output, x=intervals_comb)

            else:
                raise ValueError("A function must be provided.")

        case _: raise ValueError(
                     "Invalid UP method! focused_discretisation_cauchy is under development.")

    return results

from pyuncertainnumber import UncertainNumber
def plotPbox(xL, xR, p=None):
    """
    Plots a p-box (probability box) using matplotlib.

    Args:
        xL (np.ndarray): A 1D NumPy array of lower bounds.
        xR (np.ndarray): A 1D NumPy array of upper bounds.
        p (np.ndarray, optional): A 1D NumPy array of probabilities corresponding to the intervals.
                                   Defaults to None, which generates equally spaced probabilities.
        color (str, optional): The color of the plot. Defaults to 'k' (black).
    """
    xL = np.squeeze(xL)  # Ensure xL is a 1D array
    xR = np.squeeze(xR)  # Ensure xR is a 1D array

    if p is None:
        # p should have one more element than xL/xR
        p = np.linspace(0, 1, len(xL) + 1)

    # Plot the step functions
    plt.step(np.concatenate(([xL[0]], xL)), p, where='post', color='black')
    plt.step(np.concatenate(([xR[0]], xR)), p, where='post', color='red')

    # Add bottom and top lines to close the box
    plt.plot([xL[0], xR[0]], [0, 0], color='red')  # Bottom line
    plt.plot([xL[-1], xR[-1]], [1, 1], color='black')  # Top line

    # Add x and y axis labels
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Cumulative Probability", fontsize=14)
    # Increase font size for axis numbers
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.show()

def Fun(x):

    input1= x[0]
    input2=x[1]
    input3=x[2]
    input4=x[3]
    input5=x[4]

    output1 = input1 + input2 + input3 + input4 + input5
    output2 = input1 * input2 * input3 * input4 * input5

    return np.array([output1]) #, output2

means = np.array([1, 2, 3, 4, 5])
stds = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

x = [
    UncertainNumber(essence = 'pbox', distribution_parameters= ["gaussian",[[1,2], 0.1]]),

    UncertainNumber(essence = 'interval', bounds= [means[1]-2* stds[1], means[1]+2* stds[1]]),
    UncertainNumber(essence = 'interval', bounds= [means[2]-2* stds[2], means[2]+2* stds[2]]),
    UncertainNumber(essence = 'interval', bounds= [means[3]-2* stds[3], means[3]+2* stds[3]]),
    UncertainNumber(essence = 'interval', bounds= [means[4]-2* stds[4], means[4]+2* stds[4]])
    ]

results = interval_monte_carlo_method(x=x, f=Fun, method = 'interval_monte_carlo_genetic_optimisation', n_sam= 10)

print(results.raw_data['min'][0]['f'])
print(results.raw_data['min'][0]['x'])
print(results.raw_data['min'][0]['message'])

plotPbox(results.raw_data['min'][0]['f'], results.raw_data['max'][0]['f'], p=None)
#plotPbox(results.raw_data['min'][1]['f'], results.raw_data['max'][1]['f'], p=None)


