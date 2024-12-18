import numpy as np
import pprint
from typing import Callable
# import plotly.express as ps
from PyUncertainNumber.UP.endpoints import endpoints_method
from PyUncertainNumber.UP.endpoints_extremepoints import endpoints_extremepoints_method
from PyUncertainNumber.UP.subinterval import subinterval_method
from PyUncertainNumber.UP.sampling import sampling_method
from PyUncertainNumber.UP.genetic_optimisation import genetic_optimisation_method
from PyUncertainNumber.UP.local_optimisation import local_optimisation_method
from PyUncertainNumber.UP.endpoints_cauchy import cauchydeviates_method
from PyUncertainNumber.UP.sampling_aleatory import sampling_aleatory_method
from PyUncertainNumber.UP.utils import create_folder, save_results
from PyUncertainNumber.UC.uncertainNumber import UncertainNumber #_parse_interverl_inputs,
#TODO fix the distribution parameters if we only  have sample values. 
# ---------------------the top level UP function ---------------------#
       
def aleatory_propagation(vars:list = None,
        results:dict = None,
        fun:Callable = None,
        n_sam:int = None,
        method:str = "monte_carlo",       
        save_raw_data="no",
        *,  # Keyword-only arguments start here
        base_path=np.nan,
        **kwargs,
    ):
    
    """
    args:
        - vars (list): A list of UncertainNumber objects, each representing an input 
                    variable with its associated uncertainty.
        - fun (Callable): The function to propagate uncertainty through.
        - n (int): The number of samples to generate.
        - conf_level (float, optional): The confidence level for K-S bounds. Defaults to 0.95.
        - ks_bound_points (int, optional): The number of points to evaluate K-S bounds at. Defaults to 100.
        - method (str, optional): The sampling method ('monte_carlo' or 'latin_hypercube'). Defaults to 'monte_carlo'.
        - save_raw_data (str, optional): Whether to save raw data ('yes' or 'no'). Defaults to 'no'.
        - base_path (str, optional): Path for saving results (if save_raw_data is 'yes'). Defaults to np.nan.
        - **kwargs: Additional keyword arguments to be passed to the UncertainNumber constructor.

    signature:
        aleatory_propagation(x:np.ndarray, f:Callable, n:int, method ='montecarlo',  save_raw_data = 'no') -> dict of np.ndarrays

    note:
        - This function propagates uncertainty through a given function (`fun`) using either 
            Monte Carlo or Latin Hypercube sampling, considering the aleatory uncertainty 
            represented by a list of `UncertainNumber` objects (`vars`). 
        - It calculates Kolmogorov-Smirnov (K-S) bounds for the output(s) to provide a non-parametric 
            confidence region.
        - If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional y and x for all x samples.

    returns:
        dict: A dictionary containing:
            - 'un': A list of UncertainNumber objects, each representing the output(s)
                    of the function with K-S bounds as uncertainty.
            - 'ks_bounds': A list of dictionaries, one for each output of fun, containing the K-S bounds.
            - 'raw_data': A dictionary containing raw data (if save_raw_data is 'yes'):
                - 'x': All generated input samples.
                - 'f': Corresponding output values for each input sample.

    Raises:
        ValueError: For invalid method, save_raw_data, or missing arguments.
 
    """
    # Input validation
   
    if save_raw_data not in ("yes", "no"):  # Input validation
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")
    
    def process_alea_results(results):
        if results['raw_data']['f'] is None:
            results['un'] = UncertainNumber(essence="distribution", distribution_parameters=None, **kwargs)
        else:
            results['un'] = []
            for sample_data in results['raw_data']['f'].T:  # Transpose to iterate over columns
                results['un'].append(UncertainNumber(essence="distribution", distribution_parameters=sample_data, **kwargs))

        if save_raw_data == "yes":
            res_path = create_folder(base_path, method)
            save_results(results['raw_data'], method=method, res_path=res_path, fun=fun) # only to save raw_data as data.frames

        return results
        
    match method:
           
        case ("monte_carlo" | "latin_hypercube"): 
            if n_sam is None:
                raise ValueError("n (number of samples) is required for sampling methods.")
            results= sampling_aleatory_method(vars, fun, n_sam, method=method.lower(), results=results, save_raw_data= save_raw_data)    
            return process_alea_results(results)
        case ("taylor_expansion" ): 
            print("Taylor expansion is not implemented in this version")
        case _:
            raise ValueError("Invalid UP method.")
        
def epistemic_propagation(vars,
          fun,
          results:dict = None,
          n_sub: np.integer = None,
          n_sam: np.integer = None,
          x0: np.ndarray = None,
          method: str = None,
          save_raw_data ="no",
          *,  # Keyword-nly arguments start here
          base_path=np.nan,
          tol_loc: np.ndarray = None,
          options_loc: dict = None,
          method_loc="Nelder-Mead",
          pop_size=1000,
          n_gen=100,
          tol=1e-3,
          n_gen_last=10,
          algorithm_type="NSGA2",
          **kwargs,
          ):
    
    """Performs uncertainty propagation (UP) using various methods.

    Args:
        vars (np.ndarray or list of UN objects): Input intervals.
        fun (Callable): The function to propagate.
        n (np.integer, optional): Number of subintervals/samples. Defaults to None.
        method (str, optional): UP method. Defaults to "endpoint".
        save_raw_data (str, optional): Whether to save raw data. Defaults to "no".
        base_path (str, optional): Path for results. Defaults to np.nan.
        x0 (np.array, optional): Initial guess for local optimisation.
        objective (str, optional): Optimisation objective. Defaults to 'minimize'.
        pop_size (int, optional): Population size for genetic algorithm. Defaults to 1000.
        n_gen (int, optional): Number of generations for genetic algorithm. Defaults to 100.
        tol (float, optional): Tolerance for genetic algorithm. Defaults to 1e-3.
        n_gen_last (int, optional):  Generations for last population. Defaults to 10.
        algorithm_type (str, optional): Genetic algorithm type. Defaults to "NSGA2".

    Returns:
        tuple: Results of the UP method.

    Raises:
        ValueError: For invalid method, save_raw_data, or missing arguments.
        #TODO update the description. 
 
    """ 
    #vars = _parse_interverl_inputs(vars)
    x = np.zeros((len(vars),2))
    for i, un in enumerate(vars):
        x[i,:] = un.bounds # Get an np.array of bounds for all vars

    if method in ("local_optimisation", "genetic_optimisation"):  # Check for optimisation methods
        if not callable(fun):
            raise TypeError("fun must be a callable function for optimisation methods. fun cannot be None.")
    
    if save_raw_data not in ("yes", "no"):  # Input validation
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")
    
    def process_results(results):
        if results['raw_data']['bounds'] is None:
            results['un'] = UncertainNumber(essence="interval", bounds=None, **kwargs)
        else:
            if results['raw_data']['bounds'].ndim == 2:  # 2D array
                results['un'] = [UncertainNumber(essence="interval", bounds=bound, **kwargs) for bound in results['raw_data']['bounds']]
            elif results['raw_data']['bounds'].ndim == 1 and len(results['raw_data']['bounds']) == 2:  # 1D array
                results['un'] = UncertainNumber(essence="interval", bounds=results['raw_data']['bounds'], **kwargs)
            else:
                raise ValueError("Invalid shape for 'bounds'. Expected 2D array or 1D array with two values.")

        if save_raw_data == "yes":
            res_path = create_folder(base_path, method)
            save_results(results['raw_data'], method=method, res_path=res_path, fun=fun) # only to save raw_data as data.frames

        return results
        
    match method:
        
        case ("endpoint" | "endpoints" | "vertex"):
            results = endpoints_method(x, fun, results, save_raw_data)  # Pass save_raw_data directly
            return process_results(results) 
        
        case ("endpoints_extremepoints" ):       
            results = endpoints_extremepoints_method(x, fun,  results,save_raw_data)  # Pass save_raw_data directly
            return process_results(results) 

        case ("subinterval" | "subinterval_reconstitution"):
            if n_sub is None:
                raise ValueError("n (number of subintervals) is required for subinterval methods.")
            results = subinterval_method(x, fun, n_sub,  results, save_raw_data)  # Pass save_raw_data directly
            return process_results(results)
        
        case ("monte_carlo" |  "latin_hypercube"): 
            if n_sam is None:
                raise ValueError("n (number of samples) is required for sampling methods.")            
            results= sampling_method(x, fun, n_sam, method=method.lower(), endpoints=False,  results=results, save_raw_data= save_raw_data)    
            return process_results(results)
                
        case ("monte_carlo_endpoints" ): 
            if n_sam is None:
                raise ValueError("n (number of samples) is required for sampling methods.")           
            results= sampling_method(x, fun, n_sam, method= "monte_carlo", endpoints=True,  results=results, save_raw_data= save_raw_data)    
            return process_results(results)

        case ("latin_hypercube_endpoints" ): 
            if n_sam is None:
                raise ValueError("n (number of samples) is required for sampling methods.")           
            results= sampling_method(x, fun, n_sam, method="latin_hypercube", endpoints=True,  results=results, save_raw_data= save_raw_data)    
            return process_results(results)   

        case ("cauchy" |  "endpoint_cauchy"|  "endpoints_cauchy"): 
            if n_sam is None:
                raise ValueError("n (number of samples) is required for sampling methods.")            
            results= cauchydeviates_method(x,fun, n_sam, results, save_raw_data)
            return process_results(results)    

        case ("local_optimization" | "local_optimisation"|"local optimisation"|"local optimization") :

            if save_raw_data == 'yes':
                print("The intermediate steps cannot be saved for local optimisation")               
            results = local_optimisation_method(x, fun, x0, results=results,
                                             tol_loc = tol_loc, 
                                             options_loc = options_loc, 
                                             method_loc = method_loc)
            return process_results(results)
            
        case ("genetic_optimisation" | "genetic_optimization"| "genetic optimization"|"genetic optimisation"):
            if save_raw_data == 'yes':
                print("The intermediate steps cannot be saved for genetic optimisation")            
            results = genetic_optimisation_method(x, fun, results, pop_size, n_gen, tol, n_gen_last, 
                                               algorithm_type)                          
            return process_results(results) 
        
        case _:
            raise ValueError("Invalid UP method.")

def propagation(vars:list,
          fun:Callable,
          results:dict = None,
          n_sub: np.integer = 3,
          n_sam: np.integer = 500,
          x0: np.ndarray = None,
          method=None,
          save_raw_data="no",
          *,  # Keyword-only arguments start here
          base_path=np.nan,
          tol_loc: np.ndarray = None,
          options_loc: dict = None,
          method_loc="Nelder-Mead",
          pop_size=1000,
          n_gen=100,
          tol=1e-3,
          n_gen_last=10,
          algorithm_type="NSGA2",
          **kwargs,
          ):
    
    essences = [un.essence for un in vars]  # Get a list of all essences

    results = {
             'un': None,
           
            'raw_data': {                
                'x': None,
                'f': None,
                'min': {
                        'x': None,
                        'f': None,
                        # for genetic optimisation only
                        'message': None, #The success message from the minimization optimization.
                        'n_gen': None,
                        'n_iter': None,
                        # for local optimisation only
                        'niterations': None, #The number of iterations for minimisation.
                        'nfevaluations': None, #The number of function evaluations for minimisation.
                        'final_simplex': None #The final simplex for minimization.
                    },
                'max': {
                        'x': None,
                        'f': None,
                        # for genetic optimisation only
                        'message': None, #The success message from the minimization optimization.
                        'n_gen': None,
                        'n_iter': None,
                        # for local optimisation only
                        'niterations': None, #The number of iterations for minimisation.
                        'nfevaluations': None, #The number of function evaluations for minimisation.
                        'final_simplex': None #The final simplex for minimization.
                    },
                'bounds': None
                }
            }
 

    # Determine the plotting strategy based on essences
    if all(essence == "interval" for essence in essences):
        
        y = epistemic_propagation(vars = vars,
                                        fun = fun,
                                        n_sub = n_sub,
                                        n_sam = n_sam,
                                        x0 = x0,
                                        results= results, 
                                        method = method,
                                        save_raw_data = save_raw_data,
                                        base_path = base_path ,
                                        tol_loc= tol_loc,
                                        options_loc = options_loc,
                                        method_loc= method_loc,
                                        pop_size = pop_size, 
                                        n_gen= n_gen,
                                        tol= tol,
                                        n_gen_last=n_gen_last,
                                        algorithm_type= algorithm_type,
                                        **kwargs,
                                        )
 
    elif all(essence == "distribution" for essence in essences):
        y = aleatory_propagation(vars = vars,
                                results= results,
                                fun= fun,
                                n_sam = n_sam,
                                method= method,       
                                save_raw_data = save_raw_data,
                                base_path= base_path,
                                **kwargs)

    else:  # Mixed case or at least one p-box
        #y = mixed_propagation()
        pass

    return y 

def main():
    """ implementation of any method for epistemic uncertainty on the cantilever beam example"""

    # y = np.array([0.145, 0.155])  # m

    # L = np.array([9.95, 10.05])  # m

    # I = np.array([0.0003861591, 0.0005213425])  # m**4

    # F = np.array([11, 37])  # kN

    # E = np.array([200, 220])  # GPa

    # # Create a 2D np.array with all uncertain input parameters in the **correct** order.
    # xInt = np.array([L, I, F, E])
    
    def cantilever_beam_deflection(x):
        """Calculates deflection and stress for a cantilever beam.

        Args:
          x (np.array): Array of input parameters:
              x[0]: Length of the beam (m)
              x[1]: Second moment of area (mm^4)
              x[2]: Applied force (N)
              x[3]: Young's modulus (MPa)

      Returns:
          float: deflection (m)
                 Returns np.nan if calculation error occurs.
      """
        beam_length = x[0]
        I = x[1]
        F = x[2]
        E = x[3]
        try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
            deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m

        except:
            deflection = np.nan

        return deflection
    
   # xInt = np.array([y,L, I, F, E])
    #print(xInt)
    def cantilever_beam_func(x):
        
        y = x[0]
        beam_length = x[1]
        I = x[2]
        F = x[3]
        E = x[4]
        try:  # try is used to account for cases where the input combinations leads to error in fun due to bugs
            deflection = F * beam_length**3 / (3 * E * 10**6 * I)  # deflection in m
            stress     = F * beam_length * y / I / 1000  # stress in MPa
        
        except:
            deflection = np.nan
            stress = np.nan

        return np.array([deflection, stress])

    y = UncertainNumber(name='distance to neutral axis', symbol='y', units='m', essence='distribution', distribution_parameters=["gaussian", [0.15, 0.00333]])
    L = UncertainNumber(name='beam length', symbol='L', units='m', essence='distribution', distribution_parameters=["gaussian", [10.05, 0.033]])
    I = UncertainNumber(name='moment of inertia', symbol='I', units='m', essence='distribution', distribution_parameters=["gaussian", [0.000454, 4.5061e-5]])
    F = UncertainNumber(name='vertical force', symbol='F', units='kN', essence='distribution', distribution_parameters=["gaussian", [24, 8.67]])
    E = UncertainNumber(name='elastic modulus', symbol='E', units='GPa', essence='distribution', distribution_parameters=["gaussian", [210, 6.67]])

    # L = UncertainNumber(name='beam length', symbol='L', units='m', essence='interval', bounds= [9.95, 10.05])
    # I = UncertainNumber(name='moment of inertia', symbol='I', units='m', essence='interval', bounds= [0.0003861591, 0.0005213425])
    # F = UncertainNumber(name='vertical force', symbol='F', units='kN', essence='interval', bounds= [11, 37])
    # E = UncertainNumber(name='elastic modulus', symbol='E', units='GPa', essence='interval', bounds=[200, 220])

    METHOD = "endpoints_extremepoint"
    base_path = ""
# (vars = None,
#         results:dict = None,
#         fun = None,
#         n: np.integer = None,
#         method = "monte_carlo", 
             
#         save_raw_data="no",
#         *,  # Keyword-only arguments start here
#         base_path=np.nan,
#         **kwargs,
    a = propagation(vars= [ L, I, F, E], 
                            results = None,
                            fun= cantilever_beam_deflection, 
                           # n_sam= 5, 
                            method= METHOD, 
                            save_raw_data= "no"
                        )
    print(a['un'])


    return a

    # method = "endpoint_cauchy"
    # base_path = "C:\\Users\\Ioanna\\Documents\\GitHub\\PyUncertainNumber\\cantilever_beam"
    # #a = epistemic_propagation(xInt, fun= None, method=method, save_raw_data="yes", base_path=base_path)
    # a = epistemic_propagation(xInt, fun= None, n=10, method=method,save_raw_data="yes", base_path=base_path )
    # #pprint.pprint(a['un'])
    # #pprint.pprint(a)
    # #pprint.pprint(a)
    # print(a['un'])
    # return a

if __name__ == "__main__":
    main()
