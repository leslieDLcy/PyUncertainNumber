import numpy as np
import pprint

# import plotly.express as ps
from PyUncertainNumber.UP.endpoint import endpoints_method
from PyUncertainNumber.UP.subinterval import subinterval_method
from PyUncertainNumber.UP.sampling import sampling_method
from PyUncertainNumber.UP.genetic_optimisation import genetic_optimization_method
from PyUncertainNumber.UP.local_optimisation import local_optimisation_method
from PyUncertainNumber.UP.endpoints_cauchy import cauchydeviate_method
from PyUncertainNumber.UP.utils import post_processing, create_folder
from PyUncertainNumber.UC.uncertainNumber import _parse_interverl_inputs, UncertainNumber

# ---------------------the top level UP function ---------------------#

def up_bb(vars,
          fun,
          n: np.integer = None,
          x0: np.ndarray = None,
          method="endpoint",
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
    
    """Performs uncertainty propagation (UP) using various methods.

    Args:
        vars (np.ndarray or list of UN objects): Input intervals.
        fun (Callable): The function to propagate.
        n (np.integer, optional): Number of subintervals/samples. Defaults to None.
        method (str, optional): UP method. Defaults to "endpoint".
        save_raw_data (str, optional): Whether to save raw data. Defaults to "no".
        base_path (str, optional): Path for results. Defaults to np.nan.
        x0 (np.array, optional): Initial guess for local optimization.
        objective (str, optional): Optimization objective. Defaults to 'minimize'.
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
    # Input validation
    
    vars = _parse_interverl_inputs(vars)

    if method in ("local_optimisation", "genetic_optimisation"):  # Check for optimization methods
        if not callable(fun):
            raise TypeError("fun must be a callable function for optimization methods. fun cannot be None.")
    
    if save_raw_data not in ("yes", "no"):  # Input validation
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")
    
    def process_results(results):
        if results['bounds'] is None:
            results['un'] = UncertainNumber(essence="interval", bounds=None, **kwargs)
        else:
            if results['bounds'].ndim == 2:  # 2D array
                results['un'] = [UncertainNumber(essence="interval", bounds=bound, **kwargs) for bound in results['bounds']]
            elif results['bounds'].ndim == 1 and len(results['bounds']) == 2:  # 1D array
                results['un'] = UncertainNumber(essence="interval", bounds=results['bounds'], **kwargs)
            else:
                raise ValueError("Invalid shape for 'bounds'. Expected 2D array or 1D array with two values.")

        if save_raw_data == "yes":
            res_path = create_folder(base_path, method)
            # Handle the case where fun is None:
            if fun is None:
                Results = post_processing(results['raw_data']['x'], all_output=None, method=method, res_path=res_path)
            else:
                Results = post_processing(results['raw_data']['x'], results['raw_data'].get('f'), method, res_path)

        return results
        
    match method:
        
        case ("endpoint" | "endpoints" | "vertex"):
          
            results = endpoints_method(vars, fun, save_raw_data)  # Pass save_raw_data directly

            return process_results(results) 

        case ("subinterval" | "subinterval_reconstitution"):
            if n is None:
                raise ValueError("n (number of subintervals) is required for subinterval methods.")

            results = subinterval_method(vars, fun, n, save_raw_data)  # Pass save_raw_data directly

            return process_results(results)
        
        case ("monte_carlo" |  "latin_hypercube"): 
            if n is None:
                raise ValueError("n (number of samples) is required for sampling methods.")
            
            results= sampling_method(vars, fun, n, method=method.lower(), endpoints=False, save_raw_data= save_raw_data)    
            return process_results(results)
                
        case ("monte_carlo_endpoints" ): 
            if n is None:
                raise ValueError("n (number of samples) is required for sampling methods.")
            
            results= sampling_method(vars, fun, n, method= "monte_carlo", endpoints=True, save_raw_data= save_raw_data)    
            return process_results(results)

        case ("latin_hypercube_endpoints" ): 
            if n is None:
                raise ValueError("n (number of samples) is required for sampling methods.")
            
            results= sampling_method(vars, fun, n, method="latin_hypercube", endpoints=True, save_raw_data= save_raw_data)    
            return process_results(results)   

        case ("cauchy" |  "endpoint_cauchy"|  "endpoints_cauchy"): 
            if n is None:
                raise ValueError("n (number of samples) is required for sampling methods.")
            
            results= cauchydeviate_method(vars,fun, n, save_raw_data)

            return process_results(results)    

        case ("local_optimization" | "local_optimisation") :

            if save_raw_data == 'yes':
                print("The intermediate steps cannot be saved for local optimisation")
            results = local_optimisation_method(vars, fun, x0, 
                                             tol_loc = tol_loc, 
                                             options_loc = options_loc, 
                                             method_loc = method_loc)

            return process_results(results)
            
        case ("genetic_optimisation" | "genetic_optimization"):
            if save_raw_data == 'yes':
                print("The intermediate steps cannot be saved for genetic optimisation")
            
            results = genetic_optimization_method(vars, fun, pop_size, n_gen, tol, n_gen_last, 
                                               algorithm_type)
                          
            return process_results(results) 
        
        case _:
            raise ValueError("Invalid UP method.")

def main():
    """ (Ioanna style) implementation of the vertex method on the cantilever beam example"""

    y = np.array([0.145, 0.155])  # m

    L = np.array([9.95, 10.05])  # m

    I = np.array([0.0003861591, 0.0005213425])  # m**4

    F = np.array([11, 37])  # kN

    E = np.array([200, 220])  # GPa

    # Create a 2D np.array with all uncertain input parameters in the **correct** order.
    xInt = np.array([L, I, F, E])
    
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


    method = "endpoint_cauchy"
    base_path = "C:\\Users\\Ioanna\\Documents\\GitHub\\PyUncertainNumber\\cantilever_beam"
    #a = up_bb(xInt, fun= None, method=method, save_raw_data="yes", base_path=base_path)
    a = up_bb(xInt, fun= None, n=10, method=method,save_raw_data="yes", base_path=base_path )
    #pprint.pprint(a['un'])
    #pprint.pprint(a)
    #pprint.pprint(a)
    print(a['un'])
    return a

if __name__ == "__main__":
    main()
