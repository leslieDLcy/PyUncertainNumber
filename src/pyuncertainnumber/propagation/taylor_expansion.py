import os

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp


""" Taylor expansions for the moments of functions of random variables """


def taylor_expansion_method(func, mean, *, var=None, cov=None) -> tuple:
    """Performs uncertainty propagation using the Taylor expansion method.

    args:
        func: function to propagate uncertainty through. Expecting a iterable-signature function.
        mean (Jax array): mean of the input random variable (scalar or vector)
        var (Jax array): variance of the input random variable (scalar only)
        cov (Jax array): covariance matrix of the input random vector (vector only)

    returns:
        mu_f: mean of the output random variable through the function
        var_f: variance of the output random variable through the function

    note:
        Currently it only supports scalar-output functions. Also, for multivariate function, the
        calling signature is assumed to be func(x) where x is a 1D array, i.e. func: R^n -> R, the vec style.
        For best compatibility to work with derivatives, the `func` is better written in jax.numpy.


    example:
        >>> import jax.numpy as jnp
        >>> from pyuncertainnumber import taylor_expansion_method
        >>> MEAN= jnp.array([3., 2.5])
        >>> COV = jnp.array([[4, 0.3], [0.3, 0.25]])
        >>> def bar(x): return x[0]**2 + x[1] + 3
        >>> mu_, var_ = taylor_expansion_method(func=bar, mean=MEAN, cov=COV)

    """
    if mean.ndim == 1:  # random vector
        return taylor_expansion_method_vector(func, mean, cov)
    elif mean.ndim == 0:  # scalar random variable
        return taylor_expansion_method_scalar(func, mean, var)


def taylor_expansion_method_scalar(func, mean, var) -> tuple:
    """For scalar random variable only"""

    # gradient
    d1f = jax.grad(func)(mean)

    # second-order
    H = jax.hessian(func)
    d2f = jnp.diag(H(mean))

    mu_f = func(mean) + jnp.dot(d2f / 2.0, var)
    var_f = jnp.dot(d1f**2, var) - 1 / 4 * jnp.dot(d2f**2, var**2)

    return mu_f, var_f


def taylor_expansion_method_vector(func, mean, cov) -> tuple:
    """For random vector only"""

    # gradient
    d1f = jax.grad(func)(mean)

    # second-order
    H = jax.hessian(func)

    mu_f = func(mean) + 1 / 2 * jnp.trace(H(mean) @ cov)

    var_f = d1f @ cov @ d1f + 1 / 2 * jnp.trace(H(mean) @ cov @ H(mean) @ cov)

    return mu_f, var_f
