# import plotly.express as ps
from typing import Callable, Union
import numpy as np
from PyUncertainNumber.propagation.epistemic_uncertainty.endpoints import endpoints_method
from PyUncertainNumber.propagation.epistemic_uncertainty.subinterval import subinterval_method
from PyUncertainNumber.propagation.epistemic_uncertainty.sampling import sampling_method
from PyUncertainNumber.propagation.epistemic_uncertainty.genetic_optimisation import genetic_optimisation_method
from PyUncertainNumber.propagation.epistemic_uncertainty.local_optimisation import local_optimisation_method
from PyUncertainNumber.propagation.epistemic_uncertainty.endpoints_cauchy import cauchydeviates_method
from PyUncertainNumber.propagation.aleatory_uncertainty.sampling_aleatory import sampling_aleatory_method
from PyUncertainNumber.propagation.mixed_uncertainty.focused_discretisation_propagation import focused_discretisation_propagation_method
from PyUncertainNumber.propagation.mixed_uncertainty.varied_discretisation_propagation import varied_discretisation_propagation_method
from PyUncertainNumber.propagation.utils import create_folder, save_results, Propagation_results
from PyUncertainNumber.characterisation.uncertainNumber import UncertainNumber
from PyUncertainNumber.pba.distributions import Distribution

#TODO the cauchy with save_raw_data = 'yes' raises issues.  
# ---------------------the top level UP function ---------------------#

def aleatory_propagation(
    vars: list = None,
    fun: Callable = None,
    results: Propagation_results = None,
    n_sam: int = 500,
    method: str = "monte_carlo",
    save_raw_data="no",
    *,  # Keyword-only arguments start here
    base_path=np.nan,
    **kwargs,
):
    """This function propagates aleatory uncertainty through a given function (`fun`) using either Monte Carlo or Latin Hypercube sampling, 
       considering the aleatory uncertainty represented by a list of `UncertainNumber` objects (`vars`).
    
    args:
        vars (list): A list of UncertainNumber objects, each representing an input
                    variable with its associated uncertainty.
        fun (Callable): The function to propagate uncertainty through.
        results (Propagation_results, optional): An object to store propagation results.
                                    Defaults to None, in which case a new
                                    `Propagation_results` object is created.
        n_sam (int): The number of samples to generate.
                    Default is 500.
        method (str, optional): The sampling method ('monte_carlo' or 'latin_hypercube').
                    Defaults to 'monte_carlo'.
        save_raw_data (str, optional): Whether to save raw data ('yes' or 'no').
                    Defaults to 'no'.
        base_path (str, optional): Path for saving results (if save_raw_data is 'yes').
                    Defaults to np.nan.
        **kwargs: Additional keyword arguments to be passed to the UncertainNumber constructor.

    signature:
        aleatory_propagation(x:np.ndarray, f:Callable, n:int, ...) -> Propagation_results

    note:
        If the `f` function returns multiple outputs, the `all_output` array will be 2-dimensional y and x for all x samples.

    returns:
        Propagation_results: A  `Propagation_results` object containing:
                          - 'un': A list of UncertainNumber objects, each representing
                                  the empirical distribtuion of output(s) of the function.
                          - 'raw_data': A dictionary containing raw data (if
                                        save_raw_data is 'yes'):
                                          - 'x': All generated input samples.
                                          - 'f': Corresponding output values for each
                                                input sample.

    raises:
        ValueError: For invalid method, save_raw_data, or missing arguments.

    """
    # Input validation

    if save_raw_data not in ("yes", "no"):  # Input validation
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")

    def process_alea_results(results):
        """
        args:
            results (Propagation_results): A `Propagation_results` object containing raw
                                    epistemic propagation results. This object is
                                    modified in-place.

        signature:
            process_alea_results(results: Propagation_results) -> Propagation_results

        notes:
            Processes the results of aleatory uncertainty propagation.

            This function takes a `Propagation_results` object containing raw aleatory
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
            Propagation_results: The modified `Propagation_results` object with
                          `UncertainNumber` objects added to `results.un` and
                          potentially with raw data saved to a file.

        raises:
            ValueError: If the shape of `results.raw_data['f']` is invalid

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
                d = Distribution(sample_data= sample_data)
                results.un.append(UncertainNumber.fromDistribution(d))

        if save_raw_data == "yes":
            res_path = create_folder(base_path, method)
            save_results(results.raw_data, method=method, res_path=res_path, fun=fun)

        return results

    match method:

        case "monte_carlo" | "latin_hypercube":
            if n_sam is None:
                raise ValueError(
                    "n (number of samples) is required for sampling methods."
                )
            results = sampling_aleatory_method(
                x=vars,
                f=fun,
                results=results,
                n_sam=n_sam,
                method=method.lower(),
                save_raw_data=save_raw_data,
                **kwargs,
            )
            return process_alea_results(results)
        case "taylor_expansion":
            print("Taylor expansion is not implemented in this version")
        case _:
            raise ValueError("Invalid UP method.")

def mixed_propagation(
    vars: list = None,
    fun: Callable = None,
    results: Propagation_results = None,
    method="focused_discretisation_extremepoints",
    n_disc: Union[int, np.ndarray] = 10,
    condensation: int = None,
    tOp: Union[float, np.ndarray] = 0.999,
    bOt: Union[float, np.ndarray] = 0.001,
    save_raw_data="no",
    *,  # Keyword-nly arguments start here
    base_path=np.nan,
    **kwargs,
):
    """Performs mixed uncertainty propagation through a given function. This function handles uncertainty propagation when there's a mix of
          aleatory and epistemic uncertainty in the input variables.

    args:
        vars (list): A list of uncertain variables, which can be a mix of different
                 uncertainty types (e.g., intervals, distributions).
        fun (Callable): The function to propagate uncertainty through.
        results (Propagation_results, optional): An object to store propagation results.
                                Defaults to None, in which case a new
                                `Propagation_results` object is created.
        method (str, optional): The mixed uncertainty propagation method. Can be one of:
                                'focused_discretisation_endpoints', 'focused_discretisation_vertex', 'focused_discretisation_extremepoints',
                                'varied_discretisation_extremepoints'.
                                Defaults to 'focused_discretisation_extremepoints'.
        n_disc (Union[int, np.ndarray], optional): Number of discretization points for 
                                interval variables. 
                                Defaults to 10.
        tOp (Union[float, np.ndarray], optional): Upper threshold or bound used in some methods. 
                                Defaults to 0.999.
        bOt (Union[float, np.ndarray], optional): Lower threshold or bound used in some methods.
                                Defaults to 0.001.
        condensation (int, optional): Parameter for reducing the complexity of the output 
                                uncertainty representation. 
                                Defaults to None.
        save_raw_data (str, optional): Whether to save raw data ('yes' or 'no'). 
                                Defaults to 'no'.
        base_path (str, optional): Path for saving results (if save_raw_data is 'yes').
                               Defaults to np.nan.
        **kwargs: Additional keyword arguments passed to the underlying propagation methods.

    signature:
       mixed_propagation(vars: list, fun: Callable, results: Propagation_results = None, ...) -> Propagation_results

    notes:
        It can be used if each uncertain number is exrpessed in terms of precise distributions.

    returns:
        Propagation_results: A `Propagation_results` object containing the results of
                          the mixed uncertainty propagation. The format of the results
                          depends on the chosen `method`.

    raises:
        ValueError: For invalid `method` or `save_raw_data`.

    examples:
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
 
        a = mixed_propagation(vars= [y, L, I, F, E], 
                            fun= cantilever_beam_func, 
                            method= 'focused_discretisation_extremepoints', 
                            n_disc=8,
                            #save_raw_data= "no"#,
                            save_raw_data= "yes",
                            base_path= base_path
                        )
    """

    if save_raw_data not in ("yes", "no"):  # Input validation
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")

    def process_mixed_results(results: Propagation_results):
        """
        args:
            results (Propagation_results): A `Propagation_results` object containing raw
                                    epistemic propagation results. This object is
                                    modified in-place.

        signature:
            process_mixed_results(results: Propagation_results) -> Propagation_results

        notes:
            Processes the results of mixed uncertainty propagation.
            This function takes a `Propagation_results` object containing raw aleatory
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
            Propagation_results: The modified `Propagation_results` object with
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

        if save_raw_data == "yes":
            res_path = create_folder(base_path, method)
            save_results(results.raw_data, method=method, res_path=res_path, fun=fun) 
        
        return results
    
    match method:       
        case ("focused_discretisation_endpoints" | "focused_discretisation_vertex" | "endpoints" |"vertex"):
            results = focused_discretisation_propagation_method(vars=vars,                                
                                                fun=fun,
                                                results=results,
                                                method = method,
                                                n_disc= n_disc,
                                                
                                                tOp =tOp,
                                                bOt= bOt,
                                                condensation =condensation, 
                                                save_raw_data = save_raw_data,
                                                **kwargs,
                                                )  # Pass save_raw_data directly
            return process_mixed_results(results) 
        
        case ("focused_discretisation_extremepoints" | "extremepoints" ):       
            results =  focused_discretisation_propagation_method(vars=vars,                                
                                                fun=fun,
                                                results=results,
                                                method = method,
                                                n_disc= n_disc,
                                                tOp =tOp,
                                                bOt= bOt,
                                                condensation =condensation, 
                                                save_raw_data = save_raw_data,
                                                **kwargs,   
                                                )  # Pass save_raw_data directly
            return process_mixed_results(results) 
        
        case ("varied_discretisation"|"varied_discretisation_extremepoints"):
            results = varied_discretisation_propagation_method(vars=vars,                                
                                                fun=fun,
                                                results=results,
                                                method = method,
                                                #method = 'extremepoints',
                                                n_disc= n_disc,
                                                tOp =tOp,
                                                bOt= bOt,
                                                condensation =condensation,
                                                save_raw_data = save_raw_data,
                                                **kwargs,  
                                                ) 
            return process_mixed_results(results)
        case _:
            raise ValueError("Invalid UP method.")

def epistemic_propagation(
    vars: list = None,
    fun: Callable = None,
    results: Propagation_results = None,
    n_sub: np.integer = None,
    n_sam: np.integer = None,
    x0: np.ndarray = None,
    method: str = None,
    save_raw_data="no",
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
    """Performs epistemic uncertainty propagation through a given function. This function implements various methods for propagating epistemic uncertainty,
          typically represented as intervals.

    args:
        vars (list): A list of `UncertainNumber` objects representing the input variables
                 with their associated interval uncertainty.
        fun (Callable): The function to propagate uncertainty through.
        results (Propagation_results, optional): An object to store propagation results.
                        Defaults to None, in which case a new
                        `Propagation_results` object is created.
        n_sub (np.integer, optional): Number of subintervals for subinterval methods.
                        Defaults to None.
        n_sam (np.integer, optional): Number of samples for sampling-based methods.
                        Defaults to None.
        x0 (np.ndarray, optional): Initial guess for local optimization methods.
                        Defaults to None.
        method (str, optional): The uncertainty propagation method to use.
                        Defaults to "endpoint".
        save_raw_data (str, optional): Whether to save raw data ('yes' or 'no').
                        Defaults to "no".
        base_path (str, optional): Path for saving results (if save_raw_data is 'yes').
                        Defaults to np.nan.
        tol_loc (np.ndarray, optional): Tolerance for local optimization.
                        Defaults to None.
        options_loc (dict, optional): Options for local optimization.
                        Defaults to None.
        method_loc (str, optional): Method for local optimization.
                        Defaults to 'Nelder-Mead'.
        pop_size (int, optional): Population size for genetic algorithms.
                        Defaults to 1000.
        n_gen (int, optional): Number of generations for genetic algorithms.
                        Defaults to 100.
        tol (float, optional): Tolerance for genetic algorithms. Defaults to 1e-3.
        n_gen_last (int, optional): Number of last generations for genetic algorithms.
                      Defaults to 10.
        algorithm_type (str, optional): Type of genetic algorithm.
                        Defaults to 'NSGA2'.
        **kwargs: Additional keyword arguments passed to the `UncertainNumber` constructor.

    signature:
        epistemic_propagation(vars: list, fun: Callable, results: Propagation_results = None, ...) -> Propagation_results

    notes:
        It supports a wide range of techniques, including:

            1. Interval-based methods:
                - `endpoints` or `vertex`:  Calculates the function output at the endpoints
                                 or vertices of the input intervals.
                - `extremepoints`: Considers all possible combinations of interval endpoints
                       to find the extreme values of the output.
                - `subinterval` or `subinterval_reconstitution`: Divides the input intervals
                                                    into subintervals and performs
                                                    propagation on each subinterval.

            2. Sampling-based methods:
                - `monte_carlo`, `latin_hypercube`:  Uses Monte Carlo or Latin Hypercube
                                        sampling within the input intervals.
                - `monte_carlo_endpoints`, `latin_hypercube_endpoints`:  Combines sampling with
                                                            evaluation at interval
                                                            endpoints.
                - `cauchy`, `endpoint_cauchy`, `endpoints_cauchy`: Uses Cauchy deviates for
                                                      sampling.

            3. Optimization-based methods:
                - `local_optimization` or `local_optimisation`:  Uses local optimization
                                                      algorithms to find the minimum
                                                      or maximum output values.
                - `genetic_optimisation` or `genetic_optimization`: Uses genetic algorithms for
                                                        global optimization.
    returns:
        Propagation_results: A  `Propagation_results` object containing the results of
                          the epistemic uncertainty propagation. The format of the
                          results depends on the chosen `method`.

    raises:
        ValueError: For invalid `method`, `save_raw_data`, or missing arguments.
        TypeError: If `fun` is not callable for optimization methods.

    example:
        >>> a = epistemic_propagation(vars= [ y, L, I, F, E],
                                fun= cantilever_beam_func,
                                method= 'extremepoints',
                                n_disc=8,
                                save_raw_data= "no"
                            )
    """

    x = np.zeros((len(vars), 2))
    for i, un in enumerate(vars):
        x[i, :] = un.bounds  # Get an np.array of bounds for all vars

    if method in (
        "local_optimisation",
        "genetic_optimisation",
    ):  # Check for optimisation methods
        if not callable(fun):
            raise TypeError(
                "fun must be a callable function for optimisation methods. fun cannot be None."
            )

    if save_raw_data not in ("yes", "no"):  # Input validation
        raise ValueError("Invalid save_raw_data option. Choose 'yes' or 'no'.")

    def process_results(results: Propagation_results):
        """
        args:
            results (Propagation_results): A `Propagation_results` object containing raw
                                    epistemic propagation results. This object is
                                    modified in-place.

        notes:
            Processes the results of epistemic uncertainty propagation.
            This function takes a `Propagation_results` object containing raw epistemic
              propagation results and performs the following actions:

                1. Creates `UncertainNumber` objects:
                    - If output bounds exist in `results.raw_data['bounds']`, it creates
                     `UncertainNumber` objects with "interval" essence, representing the
                     resulting interval uncertainty.
                    - It handles both single-output (1D array of bounds) and multi-output
                     (2D array of bounds) cases.
                    - These `UncertainNumber` objects are stored in `results.un`.

                2. Saves raw data (optional):
                    - If `save_raw_data` is set to 'yes', it saves the raw propagation data
                      to a file.

        signature:
            process_results(results: Propagation_results) -> Propagation_results

        returns:
            Propagation_results: The modified `Propagation_results` object with
                          `UncertainNumber` objects added to `results.un` and
                          potentially with raw data saved to a file.

        raises:
            ValueError: If the shape of `results.raw_data['bounds']` is invalid.

        """
        if results.raw_data["bounds"] is None or results.raw_data["bounds"].size == 0:
            results.un = UncertainNumber(essence="interval", bounds=None, **kwargs)
        else:
            if results.raw_data["bounds"].ndim == 2:  # 2D array
                results.un = [
                    UncertainNumber(essence="interval", bounds=bound, **kwargs)
                    for bound in results.raw_data["bounds"]
                ]
            # 1D array
            elif (
                results.raw_data["bounds"].ndim == 1
                and len(results.raw_data["bounds"]) == 2
            ):
                results.un = UncertainNumber(
                    essence="interval", bounds=results.raw_data["bounds"], **kwargs
                )
            else:
                raise ValueError(
                    "Invalid shape for 'bounds'. Expected 2D array or 1D array with two values."
                )

        if save_raw_data == "yes":
            res_path = create_folder(base_path, method)
            save_results(results.raw_data, method=method, res_path=res_path, fun=fun)

        return results

    match method:

        case "endpoint" | "endpoints" | "vertex":
            # Pass save_raw_data directly
            results = endpoints_method(x=x, f=fun, results=results, 
                                           save_raw_data=save_raw_data)
            return process_results(results)

        case "extremepoints":
            # Pass save_raw_data directly
            results = extremepoints_method(x=x, f=fun, results=results, 
                                           save_raw_data=save_raw_data)
            return process_results(results)

        case "subinterval" | "subinterval_reconstitution":
            if n_sub is None:
                raise ValueError(
                    "n (number of subintervals) is required for subinterval methods."
                )
            # Pass save_raw_data directly
            results = subinterval_method(x=x, f=fun, results=results, n_sub=n_sub,
                                           save_raw_data=save_raw_data)
            return process_results(results)
        
        case "subinterval" | "subinterval_reconstitution":
            if n_sub is None:
                raise ValueError(
                    "n (number of subintervals) is required for subinterval methods."
                )
            # Pass save_raw_data directly
            results = subinterval_method(x=x, f=fun, results=results, n_sub=n_sub,
                                           save_raw_data=save_raw_data)
            return process_results(results)

        case "cauchy" | "endpoint_cauchy"| "endpoints_cauchy":
            if n_sam is None:
                raise ValueError(
                    "n (number of samples) is required for sampling methods."
                )
            results = cauchydeviates_method(
                x=x,
                f=fun,
                results=results,
                n_sam=n_sam,
                save_raw_data=save_raw_data
            )
            return process_results(results)
        
        case ("monte_carlo" |"latin_hypercube") :
            if n_sam is None:
                raise ValueError(
                    "n (number of samples) is required for sampling methods."
                )
            results = sampling_method(
                x=x,
                f=fun,
                results=results,
                n_sam=n_sam,
                method=method.lower(),
                endpoints= False,
                save_raw_data=save_raw_data,
            )
            return process_results(results)
        
        case "monte_carlo_endpoints" :
            if n_sam is None:
                raise ValueError(
                    "n (number of samples) is required for sampling methods."
                )
            results = sampling_method(
                x=x,
                f=fun,
                results=results,
                n_sam=n_sam,
                method='monte_carlo',
                endpoints=True,
                save_raw_data=save_raw_data,
            )
            return process_results(results)
        
        case "latin_hypercube_endpoints":
            if n_sam is None:
                raise ValueError(
                    "n (number of samples) is required for sampling methods."
                )
            results = sampling_method(
                x=x,
                f=fun,
                results=results,
                n_sam=n_sam,
                method='latin_hypercube',
                endpoints=True,
                save_raw_data=save_raw_data,
            )
            return process_results(results)
            
        case ("genetic_optimisation" | "genetic_optimization"| "genetic optimization"|"genetic optimisation"):
            if save_raw_data == 'yes':
                print("The intermediate steps cannot be saved for genetic optimisation methods")            
            results = genetic_optimisation_method(x=x, f=fun, results=results, 
                                                  pop_size=pop_size, n_gen=n_gen, tol= tol, n_gen_last= n_gen_last, 
                                                  algorithm_type= algorithm_type) 
            print('results', results)                     
            return process_results(results) 
        
        case ("local_optimisation" | "local_optimization"):
            if save_raw_data == 'yes':
                print("The intermediate steps cannot be saved for local optimisation methods")            
            results = local_optimisation_method(x=vars, f=fun,  results=results, 
                              x0 = x0, tol_loc = tol_loc, options_loc = options_loc, method_loc = method_loc)
            print('results', results)                     
            return process_results(results) 
        
        case _:
            raise ValueError("Invalid UP method.")

def Propagation(vars:list,
        fun:Callable,
        results: Propagation_results = None,
        n_sub: np.integer = 3,
        n_sam: np.integer = 500,
        x0: np.ndarray = None,
        method=None,
        n_disc: Union[int, np.ndarray] = 10, 
        
        tOp: Union[float, np.ndarray] = 0.999,
        bOt: Union[float, np.ndarray] = 0.001,
        condensation:int = None,
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
          **kwargs
          ):
    """ 
    args:
        vars (list): A list of uncertain variables.
        fun (Callable): The function through which to propagate uncertainty.
        results (Propagation_results, optional): An object to store propagation results.
                                Defaults to None, in which case a new
                                `Propagation_results` object is created.
        n_sub (np.integer, optional): Number of subintervals for interval-based methods.
                            Defaults to 3.
        n_sam (np.integer, optional): Number of samples for Monte Carlo simulation.
                            Defaults to 500.
        x0 (np.ndarray, optional): Initial guess for optimization-based methods.
                            Defaults to None.
        method (str, optional):  Specifies the uncertainty propagation method.
                            Defaults to None, which triggers automatic selection.
        n_disc (Union[int, np.ndarray], optional): Number of discretization points.
                            Defaults to 10.
        tOp (Union[float, np.ndarray], optional): Upper threshold or bound. 
                            Defaults to 0.999.
        bOt (Union[float, np.ndarray], optional): Lower threshold or bound.
                            Defaults to 0.001.
        condensation (int, optional): Parameter for reducing output complexity. 
                            Defaults to None.
        save_raw_data (str, optional): Whether to save intermediate results ('yes' or 'no'). 
                            Defaults to 'no'.
        base_path (str, optional): Path for saving data. Defaults to np.nan.
        tol_loc (np.ndarray, optional): Tolerance for local optimization.
                            Defaults to None.
        options_loc (dict, optional): Options for local optimization.
                            Defaults to None.
        method_loc (str, optional): Method for local optimization.
                            Defaults to 'Nelder-Mead'.
        pop_size (int, optional): Population size for genetic algorithms.
                            Defaults to 1000.
        n_gen (int, optional): Number of generations for genetic algorithms.
                           Defaults to 100.
        tol (float, optional): Tolerance for genetic algorithms. Defaults to 1e-3.
        n_gen_last (int, optional): Number of last generations for genetic algorithms.
                            Defaults to 10.
        algorithm_type (str, optional): Type of genetic algorithm.
                            Defaults to 'NSGA2'.
        **kwargs: Additional keyword arguments passed to the underlying propagation methods.

    signature:
       Propagation(vars: list, fun: Callable, results: Propagation_results = None, ...) -> Propagation_results


    return:
        Propagation_results: A  `Propagation_results` object including:
                        - 'un': A list of UncertainNumber objects, each representing
                                  the output(s) of the function.
                        - 'raw_data': depending on the method selected.

    example:

        >>> a = Propagation(vars= [ y, L, I, F, E],
                            fun= cantilever_beam_func,
                            method= 'extremepoints',
                            n_disc=8,
                            save_raw_data= "no"
                        )
    """
    essences = [un.essence for un in vars]  # Get a list of all essences

    if results is None:
        results = Propagation_results()  # Create an instance of Propagation_results

    # Determine the plotting strategy based on essences
    if all(essence == "interval" for essence in essences):
        
        y = epistemic_propagation(vars = vars,
                                        fun = fun,
                                        results= results, 
                                        n_sub = n_sub,
                                        n_sam = n_sam,
                                        x0 = x0,
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
        if method in ("focused_discretisation_endpoints", "focused_discretisation_vertex", 
                      "focused_discretisation_extremepoints", "varied_discretisation_extremepoints"):
            y = mixed_propagation(vars=vars, 
                                  fun=fun, 
                                  results=results,
                                  method = method, 
                                  n_disc=n_disc,                                 
                                  tOp=tOp,
                                  bOt=bOt,
                                  condensation=condensation, 
                                  save_raw_data=save_raw_data,
                                  base_path=base_path,
                                  **kwargs)
        else:  # Use aleatory propagation if method is not in the list above
            y = aleatory_propagation(vars=vars,
                                      fun=fun,
                                      results=results,
                                      n_sam=n_sam,
                                      method=method,
                                      save_raw_data=save_raw_data,
                                      base_path=base_path,
                                      **kwargs)
    else:  # Mixed case or at least one p-box
        y = mixed_propagation(vars = vars,                                
                              fun= fun,
                              results= results,
                             method = method, 
                              n_disc= n_disc,
                              tOp =tOp,
                              bOt= bOt,
                              condensation =condensation, 
                              save_raw_data = save_raw_data,
                              base_path= base_path,
                                **kwargs)                               

    return y

# def plotPbox(xL, xR, p=None):
#     """
#     Plots a p-box (probability box) using matplotlib.

#     Args:
#         xL (np.ndarray): A 1D NumPy array of lower bounds.
#         xR (np.ndarray): A 1D NumPy array of upper bounds.
#         p (np.ndarray, optional): A 1D NumPy array of probabilities corresponding to the intervals.
#                                    Defaults to None, which generates equally spaced probabilities.
#         color (str, optional): The color of the plot. Defaults to 'k' (black).
#     """
#     xL = np.squeeze(xL)  # Ensure xL is a 1D array
#     xR = np.squeeze(xR)  # Ensure xR is a 1D array

#     if p is None:
#         p = np.linspace(0, 1,len(xL))  # p should have one more element than xL/xR

#     if p.min() > 0:
#         p = np.concatenate(([0], p))
#         xL = np.concatenate(([xL[0]], xL))
#         xR = np.concatenate(([xR[0]], xR))

#     if p.max() < 1:
#         p = np.concatenate((p, [1]))
#         xR = np.concatenate((xR, [xR[-1]]))
#         xL = np.concatenate((xL, [xL[-1]]))
    
#     colors = 'black'
#     # Highlight the points (xL, p)
#     plt.scatter(xL, p, color=colors, marker='o', edgecolors='black', zorder=3)

#     # Highlight the points (xR, p)
#     plt.scatter(xR, p, color=colors, marker='o', edgecolors='black', zorder=3)

#     plt.fill_betweenx(p, xL, xR, color=colors, alpha=0.5)
#     plt.plot( [xL[0], xR[0]], [0, 0],color=colors, linewidth=3)
#     plt.plot([xL[-1], xR[-1]],[1, 1],  color=colors, linewidth=3)
#     plt.show()

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

        return np.array([deflection])
    
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
#     # example
    y = UncertainNumber(name='distance to neutral axis', symbol='y', units='m', essence='distribution', distribution_parameters=["gaussian", [0.15, 0.00333]])
    L = UncertainNumber(name='beam length', symbol='L', units='m', essence='distribution', distribution_parameters=["gaussian", [10.05, 0.033]])
    I = UncertainNumber(name='moment of inertia', symbol='I', units='m', essence='distribution', distribution_parameters=["gaussian", [0.000454, 4.5061e-5]])
    F = UncertainNumber(name='vertical force', symbol='F', units='kN', essence='distribution', distribution_parameters=["gaussian", [24, 8.67]])
    E = UncertainNumber(name='elastic modulus', symbol='E', units='GPa', essence='distribution', distribution_parameters=["gaussian", [210, 6.67]])
    
    # y = UncertainNumber(name='beam width', symbol='y', units='m', essence='interval', bounds=[0.145, 0.155]) 
    # L = UncertainNumber(name='beam length', symbol='L', units='m', essence='interval', bounds= [9.95, 10.05])
    # I = UncertainNumber(name='moment of inertia', symbol='I', units='m', essence='interval', bounds= [0.0003861591, 0.0005213425])
    # F = UncertainNumber(name='vertical force', symbol='F', units='kN', essence='interval', bounds= [11, 37])
    # E = UncertainNumber(name='elastic modulus', symbol='E', units='GPa', essence='interval', bounds=[200, 220])
  
    METHOD = "extremepoints"
    base_path = "C:\\Users\\Ioanna\\OneDrive - The University of Liverpool\\DAWS2_code\\UP\\"

    a = Propagation(vars= [ L, I, F, E], 
                            fun= cantilever_beam_deflection, 
                            method= 'latin_hypercube', 
                            n_sam=8,
                            #save_raw_data= "no"#,
                            save_raw_data= "yes",
                            base_path= base_path
                        )
    
    a.summary()
    # plotPbox(a.raw_data['min'][0]['f'], a.raw_data['max'][0]['f'], p=None)
    # plt.show()
    # plotPbox(a.raw_data['min'][1]['f'], a.raw_data['max'][1]['f'], p=None)
    # plt.show() 
   
    return a

if __name__ == "__main__":
    main()
