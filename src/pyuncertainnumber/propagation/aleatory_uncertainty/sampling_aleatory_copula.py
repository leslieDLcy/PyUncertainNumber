import numpy as np
from typing import Callable
import openturns as ot  # Import OpenTURNS
from pyuncertainnumber.propagation.utils import Propagation_results

def monte_carlo_copula(x: list,
                        f: Callable,
                        results: Propagation_results = None,
                        n_sam: int = 500,
                        method:str = "monte_carlo",
                        save_raw_data:str ="no", 
                        copula_type = None,
                        **kwargs):
    """
    Performs Monte Carlo simulation with copula-dependent input variables,
    using UncertainNumber objects for marginal distributions.

    Args:
        x: A list of UncertainNumber objects representing the input variables.
        f: The function to evaluate (optional). It should take the input variables as arguments.
        results: A Propagation_results object to store the results (optional).
        n_sam: The number of Monte Carlo samples.
        method: The uncertainty propagation method (currently only "monte_carlo" is supported).
        save_raw_data: Whether to save the raw transformed samples ('yes' or 'no').
        copula_type: The type of copula (e.g., "clayton", "gumbel", "frank", "gaussian", "student").
        **kwargs: Keyword arguments for copula parameters:
            - theta: Parameter for Archimedean copulas.
            - corr_matrix: Correlation matrix for Gaussian and Student copulas.
            - df: Degrees of freedom for Student copula.

    Returns:
        A Propagation_results object containing the results.
    """
    if results is None:
        results = Propagation_results()  # Create a default Propagation_results object
    
    dim = len(x)

    # Create copula object based on copula_type
    if copula_type.lower() == "clayton":
        copula = ot.ClaytonCopula(kwargs.get('theta'))  # Use OpenTURNS copula
    elif copula_type.lower() == "gumbel":
        copula = ot.GumbelCopula(kwargs.get('theta'), dim)
    elif copula_type.lower() == "frank":
        copula = ot.FrankCopula(kwargs.get('theta'), dim)
    elif copula_type.lower() == "gaussian":
        # Convert NumPy correlation matrix to OpenTURNS CorrelationMatrix
        corr_matrix_ot = ot.CorrelationMatrix(kwargs.get('corr_matrix').tolist())
        copula = ot.NormalCopula(corr_matrix_ot)
    elif copula_type.lower() == "student":
        # Convert NumPy correlation matrix to OpenTURNS CorrelationMatrix
        corr_matrix_ot = ot.CorrelationMatrix(kwargs.get('corr_matrix').tolist())
        copula = ot.StudentCopula(corr_matrix_ot, kwargs.get('df'))
    else:
        raise ValueError("Invalid copula_type.")

    # Generate copula samples
    copula_samples = copula.getSample(n_sam)
    print('copula_samples',  type(copula_samples))
    print('copula_samples',  copula_samples[:,1])

    # Transform to marginal distributions using UncertainNumber's essence distribution
    transformed_samples = {}
    for i, uncertain_num in enumerate(x):
        transformed_samples[f"x{i+1}"] = uncertain_num.ppf(copula_samples[:, i])

    # Evaluate the function if provided
    if f is not None:
        args = [transformed_samples[f"x{i+1}"] for i in range(dim)]
        results.add_raw_data(x= args)
        results.add_raw_data(f= f(*args))
    elif save_raw_data.lower() == "yes":  # If f is None and save_raw_data is 'yes'
        results.add_raw_data(x=np.array([transformed_samples[f"x{i+1}"] for i in range(dim)]))  # Store transformed samples
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")

    return results

from pyuncertainnumber.characterisation.uncertainNumber import UncertainNumber

# Define the function to evaluate
def my_function(x):
    # Example function with multiple outputs
    out1 = x[0]**2 + 2*x[1]*x2
    out2 = np.sin(x[1]) + np.log(x[2] + 1)
    return np.array([out1, out2])  # Return as a NumPy array

# Define UncertainNumber objects with essence distributions
x1 = UncertainNumber(name='distance to neutral axis', symbol='y', units='m', essence='distribution', distribution_parameters=["gaussian", [0.15, 0.00333]])
x2 = UncertainNumber(name='beam length', symbol='L', units='m', essence='distribution', distribution_parameters=["gaussian", [10.05, 0.033]])
x3 = UncertainNumber(name='moment of inertia', symbol='I', units='m', essence='distribution', distribution_parameters=["gaussian", [0.000454, 4.5061e-5]])
 
# Choose a copula and parameters
copula_type = "clayton"
theta = 3.0

# Run Monte Carlo simulation with the function
results_with_function = monte_carlo_copula(
    x=[x1, x2, x3],
    f=my_function,
    copula_type=copula_type,
    theta=theta,
    n_sam=10,
    save_raw_data="yes"
)

# Analyze results
print("Results with function:")
print("Mean of outputs:", np.mean(results_with_function.f, axis=0))
print("Variance of outputs:", np.var(results_with_function.f, axis=0))

# Run Monte Carlo simulation without the function (only generate samples)
results_no_function = monte_carlo_copula(
    x=[x1, x2, x3],
    f=None,
    copula_type=copula_type,
    theta=theta,
    n_sam=10000,
    save_raw_data="yes"
)


# Access the raw transformed samples
transformed_samples = results_no_function.raw_data

# Analyze or use the transformed samples
print("\nShape of transformed_samples:", transformed_samples.shape)
#... further analysis or use of transformed_samples