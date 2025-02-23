import numpy as np
import matplotlib.pyplot as plt
from clayton.rng import base, evd, archimedean, monte_carlo
from pyDOE2 import lhs
from scipy.stats import norm, expon  # For marginal distributions

def generate_copula_samples(copula, n_samples, marginal_distributions, sampling_method="lhs", copula_params=None, correlation_matrix=None):
    """
    Generates copula samples using Monte Carlo.

    Args:
        copula: A copula object.
        n_samples: The number of samples to generate.
        marginal_distributions: A dictionary specifying the marginal distributions.
        sampling_method: Either "lhs" (for Latin Hypercube) or "mc" (for Monte Carlo).
        copula_params (optional): A dictionary of copula parameters (e.g., {"theta": 2.0}).
        correlation_matrix (optional): A correlation matrix (for multivariate copulas).

    Returns:
        A dictionary where keys are variable names and values are the generated samples,
        or None if there is an error.
    """

    n_dimensions = len(marginal_distributions)

    # Copula Initialization (same as before)
    if correlation_matrix is not None:
        #... (same logic as before for fitting with correlation matrix)
        if copula_params is not None:
            print("Error: Provide either correlation_matrix OR copula_params, not both.")
            return None
        try:
            copula = copula.fit(correlation_matrix)
        except AttributeError:
            print("Error: Copula object does not have the fit method")
            return None
        except Exception as e:
            print(f"Error fitting copula: {e}")
            return None

    elif copula_params is not None:  # If copula parameters are provided
        try:
            copula = copula(**copula_params)  # Create copula with parameters
        except TypeError as e:
            print(f"Error initializing copula with parameters: {e}")
            return None
        except Exception as e:
            print(f"Error initializing copula: {e}")
            return None

    elif hasattr(copula, 'fit'):  # If the copula object has the fit method but no correlation matrix nor copula parameters are provided
        print("Error: Copula object requires a correlation matrix or copula parameters but none were provided")
        return None

    # Sampling
    if sampling_method == "lhs":
        u = lhs(n_dimensions, samples=n_samples, criterion='center')
    elif sampling_method == "mc":
        u = copula.sample(n_samples)  # Monte Carlo sampling
    else:
        print("Error: Invalid sampling method. Choose 'lhs' or 'mc'.")
        return None

    # Transform to copula samples (only needed for LHS, MC samples are already copula samples)
    if sampling_method == "lhs":
        try:
            v = copula.icdf(u)
        except AttributeError:
            print("Error: Copula object does not have the icdf method")
            return None
        except Exception as e:
            print(f"Error getting inverse CDF: {e}")
            return None
    else:  # Monte Carlo already has copula samples
        v = u

    # Transform to marginal distributions (same as before)
    samples = {}
    for i, (var_name, (dist_func, *params)) in enumerate(marginal_distributions.items()):
        samples[var_name] = dist_func(v[:, i], *params)

    return samples


import numpy as np
import matplotlib.pyplot as plt
from copulas.multivariate import GumbelCopula
from scipy.stats import norm, expon

# 1. Define the Gumbel copula
theta = 2.0  # Parameter for the Gumbel copula (controls dependence)
dim = 2      # Dimension of the copula (bivariate in this case)
gumbel_copula = GumbelCopula(theta, dim=dim)

# 2. Generate samples from the Gumbel copula
n_samples = 1000
samples = gumbel_copula.sample(n_samples)

# 3. Visualize the copula samples
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6)
plt.title(f"Gumbel Copula Samples (theta={theta})")
plt.xlabel("U1")
plt.ylabel("U2")
plt.grid(True)
plt.show()

# 4. Define marginal distributions
marginal_distributions = {
    "x1": (norm.ppf, 0, 1),  # Normal(0, 1)
    "x2": (expon.ppf, 2, 0.5), # Exponential(loc=2, scale=0.5)
}

# 5. Transform copula samples to marginal distributions
transformed_samples = {}
for i, (var_name, (dist_func, *params)) in enumerate(marginal_distributions.items()):
    transformed_samples[var_name] = dist_func(samples[:, i], *params)

# 6. Visualize transformed samples
plt.figure(figsize=(12, 5))
for i, (var_name, samples) in enumerate(transformed_samples.items()):
    plt.subplot(1, 2, i + 1)
    plt.hist(samples, bins=30, alpha=0.7)
    plt.title(f"{var_name} (Marginal Distribution)")
    plt.grid(True)
plt.show()


# 7. Function to generate copula samples with marginal distributions
def generate_copula_samples(copula, n_samples, marginal_distributions):
    """
    Generates copula samples and transforms them to marginal distributions.
    """
    samples = copula.sample(n_samples)
    transformed_samples = {}
    for i, (var_name, (dist_func, *params)) in enumerate(marginal_distributions.items()):
        transformed_samples[var_name] = dist_func(samples[:, i], *params)
    return transformed_samples

# Example usage
samples_gumbel = generate_copula_samples(gumbel_copula, n_samples, marginal_distributions)

# Visualize the transformed samples (similar to step 6)
#...

# sample = copula.sample(n_samples, inv_cdf= [(norm.ppf, 0, 1), (expon.ppf, 2, 0.5), (norm.ppf, 5, 2)]) #Explicitly added inv_cdf=[]

# fig, ax = plt.subplots()
# ax.scatter(sample[:, 0], sample[:, 1],
#             edgecolors='#6F6F6F', color='#C5C5C5', s=5)
# ax.set_xlabel(r'$u_0$')
# ax.set_ylabel(r'$u_1$')
# plt.show()


