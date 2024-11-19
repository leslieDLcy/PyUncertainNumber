import numpy as np

# import plotly.express as ps
from .endpoint import endpoints_method
from .subinterval_old import subinterval_method
from .sampling_old import sampling_method
from .genetic_optimisation import genetic_optimization_method
from .local_optimisation import local_optimisation_method
from .cauchy_old import cauchydeviate_method
from .utils import post_processing, create_folder
from ..UC.uncertainNumber import _parse_interverl_inputs, UncertainNumber

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
          objective='minimize',
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
        #TODO update the genetic optimisation technique to do both minimisation nad optimisation at the same time. 
        # TODO return either as namedTuple or dict to be explicit
    """
    # Input validation
    
    vars = _parse_interverl_inputs(vars)

    if not callable(fun):
        raise ValueError("f must be a callable function")

    match method:
        # TODO COmbine the calling signature of {"endpoints", 'subintervals' and 'sampling'} as they are almost the same
   
        case ("endpoint" | "endpoints" | "vertex"):
            # TODO integrate the below into the one 'endpoints' function
            # naive imple of hint on combination number of endpoints method
            print(f"Total number of input combinations for the endpoint method: {(vars.shape[0])**2}") 

            if save_raw_data == "no":
                return endpoints_method(vars, fun, save_raw_data='no')
            elif save_raw_data == "yes":
                # returned results
                min_candidate, max_candidate, x_miny, x_maxy, all_input, all_output = endpoints_method(vars, fun, save_raw_data='yes')

                un =  UncertainNumber(
                    essence="interval",
                    bounds=(min_candidate, max_candidate),
                    **kwargs,
                )

                # saving to machine part
                res_path = create_folder(base_path, method)
                Results = post_processing(all_input, all_output, res_path)  # Assuming these functions are defined elsewhere

                return {
                        'UN': un,
                        'lo_x': x_miny, 
                        'hi_x': x_maxy, 
                        'all_input': all_input, 
                        'all_output': all_output
                        }
            else:
                raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")


        case ("subinterval" | "subinterval_reconstitution"):
            if n is None:
                raise ValueError("n (number of subintervals) is required for subinterval methods.")
            if n is not None:
                print(f"Total number of input combinations for the subinterval method: {(n+1)**vars.shape[0]}")

            if save_raw_data == "no":
                return subinterval_method(vars, fun, n, save_raw_data='no')
            elif save_raw_data == "yes":
                min_candidate, max_candidate, x_miny, x_maxy, all_input, all_output = subinterval_method(vars, fun, n, save_raw_data='yes')
                
                un =  UncertainNumber(
                    essence="interval",
                    bounds=(min_candidate, max_candidate),
                    **kwargs,
                )

                res_path = create_folder(base_path, method)
                Results = post_processing(all_input, all_output, res_path)
                return {
                        'UN': un,
                        'lo_x': x_miny, 
                        'hi_x': x_maxy, 
                        'all_input': all_input, 
                        'all_output': all_output
                        }
            else:
                raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")


        case ("sampling" | "montecarlo" | "MonteCarlo" | "monte_carlo" | "latin_hypercube" | "latinhypercube" | "lhs"):
            if n is None:
                raise ValueError("n (number of samples) is required for sampling methods.")
            if save_raw_data == "no":
                return sampling_method(vars, fun, n, method=method.lower(), save_raw_data='no')
            elif save_raw_data == "yes":
                min_candidate, max_candidate, x_miny, x_maxy, all_input, all_output = sampling_method(vars, fun, n, method=method.lower(), save_raw_data='yes')

                un =  UncertainNumber(
                    essence="interval",
                    bounds=(min_candidate, max_candidate),
                    **kwargs,
                )

                res_path = create_folder(base_path, method)
                Results = post_processing(all_input, all_output, res_path)
                return {
                        'UN': un,
                        'lo_x': x_miny, 
                        'hi_x': x_maxy, 
                        'all_input': all_input, 
                        'all_output': all_output
                        }
            else:
                raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")


        case "local_optimisation":
            if not callable(fun):
                raise TypeError("fun must be a callable function for optimization methods. It cannot be None.")
            if not save_raw_data:
                print("The intermediate steps cannot be saved for local optimisation")
            return local_optimisation_method(vars, 
                                             fun, 
                                             x0, 
                                             tol_loc=tol_loc, 
                                             options_loc=options_loc, 
                                             method_loc=method_loc
                                             )
            


        case "genetic_optimisation":
            if not callable(fun):
                raise TypeError("fun must be a callable function for optimization methods. It cannot be None.")
            optimized_f, optimized_x, number_of_generations, number_of_iterations = genetic_optimization_method(vars, 
                                                                                                                fun, 
                                                                                                                objective, 
                                                                                                                pop_size,
                                                                                                                n_gen, 
                                                                                                                tol, 
                                                                                                                n_gen_last,
                                                                                                                algorithm_type
                                                                                                                )
            return optimized_f, optimized_x, number_of_generations, number_of_iterations


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
    print(xInt)

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

    # perhaps we need to add a new input to the function with the names of the input output in line with UN code
    # check the code with the xfoil code
    # a = vertexMethod(intervals, fun)
    # df_OUTPUT_INPUT = subintervalMethod(intervals, fun, n=2)

    method = "local_optimisation"
    # base_path = "C:\\Users\\Ioanna\\Documents\\GitHub\\daws2\\cantilever_beam"
    a = up_bb(xInt, cantilever_beam_deflection, x0=None, method=method, save_raw_data="no")

    print(a)
    return a


if __name__ == "__main__":
    main()
