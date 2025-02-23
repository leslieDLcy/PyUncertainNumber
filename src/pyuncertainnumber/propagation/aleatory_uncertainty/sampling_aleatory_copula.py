import numpy as np
from typing import Callable
import tqdm
import openturns as ot  # Import OpenTURNS
from pyuncertainnumber.propagation.utils import Propagation_results
#TODO as it stands it can only deal with bivariate depedence for Clayton, Frank and Gumbel
def monte_carlo_copula( x: list,
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
        copula = ot.GumbelCopula(kwargs.get('theta'))
    elif copula_type.lower() == "frank":
        copula = ot.FrankCopula(kwargs.get('theta'))
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
    copula_samples = np.array(copula_samples)  # Convert to NumPy array

    # Pre-allocate transformed_samples array
    transformed_samples = np.zeros((dim, n_sam))

    # Transform to marginal distributions using UncertainNumber's essence distribution
    for i, uncertain_num in enumerate(x):
        transformed_samples[i,:] = uncertain_num.ppf(copula_samples[:, i])  # Assign to pre-allocated array

    # Evaluate the function if provided
    if f is not None:
        all_output = np.array(
            [f(xi) for xi in tqdm.tqdm(transformed_samples.T, desc="Evaluating samples")]
        )

        if all_output.ndim == 1:  # If f returns a single output
            # Reshape to a column vector
            all_output = all_output.reshape(-1, 1)

        # Transpose transformed_samples to have each row as a sample
        results.add_raw_data(x = transformed_samples.T)
        results.add_raw_data(f = all_output)
    elif save_raw_data.lower() == "yes":
        # If f is None and save_raw_data is 'yes'
        results.add_raw_data(x=transformed_samples.T)
    else:
        print("No function is provided. Select save_raw_data = 'yes' to save the input combinations")
    return results

from pyuncertainnumber.characterisation.uncertainNumber import UncertainNumber

# Define the function to evaluate
def my_function(x):
    # Example function with multiple outputs
    out1 = x[0]**2 + 2*x[1]
    out2 = np.sin(x[1]) + np.log(x[0] + 1)
    return np.array([out1, out2])  # Return as a NumPy array

# Define UncertainNumber objects with essence distributions
x1 = UncertainNumber(name='distance to neutral axis', symbol='y', units='m', essence='distribution', distribution_parameters=["gaussian", [0.15, 0.00333]])
x2 = UncertainNumber(name='beam length', symbol='L', units='m', essence='distribution', distribution_parameters=["gaussian", [10.05, 0.033]])
x3 = UncertainNumber(name='moment of inertia', symbol='I', units='m', essence='distribution', distribution_parameters=["gaussian", [0.000454, 4.5061e-5]])
 
# Choose a copula and parameters
copula_type = "clayton"
theta = 3.0

# Analyze results
print("Results with function:")
#print("Mean of outputs:", np.mean(results_with_function.f, axis=0))
#print("Variance of outputs:", np.var(results_with_function.f, axis=0))

# Run Monte Carlo simulation without the function (only generate samples)
results_with_function = monte_carlo_copula(
    x=[x1, x2],
    f=my_function,
    copula_type=copula_type,
    theta=theta,
    n_sam=10,
    save_raw_data="yes"
)


# Access the raw transformed samples
print('x', results_with_function.raw_data['x'])
print('f', results_with_function.raw_data['f'])


#... further analysis or use of transformed_samples