"""
Test file for `tmcmc` implementation on a simple example of 2DOF system
"""

from numpy.typing import ArrayLike
import numpy as np
from pyuncertainnumber.calibration import pdfs
from pyuncertainnumber.calibration.tmcmc import TMCMC
from pyuncertainnumber import pba

# measurement data:
# eigen values of first mode
data1 = np.array([0.3860, 0.3922, 0.4157, 0.3592, 0.3615])
# eigen values of second mode
data2 = np.array([2.3614, 2.5877, 2.7070, 2.3875, 2.7272])
# eigen vector of first mode
data3 = np.array([1.68245252, 1.71103903, 1.57876073, 1.58722342, 1.61878479])

# number of particles (to approximate the posterior)
N = 100

# prior distribution of parameters
k1 = pdfs.Uniform(lower=0.8, upper=2.2)
k2 = pdfs.Uniform(lower=0.4, upper=1.2)

# Required! a list of all parameter objects
all_pars = [k1, k2]

k1 = pba.Distribution("uniform", (0.8, 2.2))
k2 = pba.Distribution("uniform", (0.4, 1.2))

# Required! a list of all parameter objects
all_pars = [k1, k2]


def log_likelihood_case3(particle_num: int, s: ArrayLike) -> float:
    """
    Required!

    log-likelihood function which is problem specific
    for the 2DOF example log-likelihood is

    args
        particle_num (int): int, particle number.

        s (ArrayLike) : numpy array of size Nop (number of parameters in all_pars)
            parameter vector of one particle in parameter space of dimension Nop, i.e. the particle's coordinates in parameter space.

    returns
        LL (float): log-likelihood function value.

    """
    sig1 = 0.0191
    sig2 = 0.0809  # = 0.05*1.618
    lambda1_s = (s[0] / 2 + s[1]) - np.sqrt(((s[0] / 2 + s[1]) ** 2 - s[0] * s[1]))
    phi12_s = (s[0] + s[1] - lambda1_s) / s[1]

    # see slide 21 -- case 3 of lecture notes
    LL = (
        np.log((2 * np.pi * sig1 * sig2) ** -5)
        + (-0.5 * (sig1 ** (-2)) * sum((lambda1_s - data1) ** 2))
        + (-0.5 * (sig2**-2) * sum((phi12_s - data3) ** 2))
    )
    return LL


def log_likelihood_case2(particle_num, s):
    """
    Log-likelihood for the 2DOF example - CASE 2 (two eigenvalues λ1 and λ2).

    Args:
        particle_num (int):
            Index of the particle (not used in this function, but required by the TMCMC framework).

        s (ArrayLike): numpy array of shape (2,)
            Parameter vector in this case:
                s[0] = q1 (stiffness parameter 1)
                s[1] = q2 (stiffness parameter 2)

    returns:
        LL (float): Log-likelihood value at this parameter vector.
    """
    q1 = s[0]
    q2 = s[1]

    # standard deviations (5% noise on true eigenvalues λ1=0.382, λ2=2.618)
    sig1 = 0.0191  # 0.05 * 0.382
    sig2 = 0.1309  # 0.05 * 2.618

    # compute eigenvalues λ1(q1, q2) and λ2(q1, q2) for the 2×2 system
    # characteristic equation: λ^2 - (q1 + 2 q2) λ + q1 q2 = 0
    # closed-form solution:
    # λ₁,₂ = (q1/2 + q2) ∓ sqrt((q1/2 + q2)^2 - q1 q2)
    center = q1 / 2.0 + q2
    disc = center**2 - q1 * q2
    if disc < 0:
        # if the discriminant is negative, eigenvalues are complex -> impossible here physically
        # give a very low likelihood
        return -np.inf

    sqrt_disc = np.sqrt(disc)
    lambda1_s = center - sqrt_disc
    lambda2_s = center + sqrt_disc

    # Gaussian likelihood for 5 measurements of λ1 and 5 of λ2
    # p(d | θ) ∝ exp( -1/(2σ1²) Σ (λ1_s - d1_m)² - 1/(2σ2²) Σ (λ2_s - d2_m)² )
    # log p(d | θ) = const - 0.5/σ1² Σ (λ1_s - d1_m)² - 0.5/σ2² Σ (λ2_s - d2_m)²

    # constant term (same form as in your Case 3 implementation)
    const_term = np.log((2 * np.pi * sig1 * sig2) ** -5)

    # misfit terms
    misfit1 = -0.5 * (sig1**-2) * np.sum((lambda1_s - data1) ** 2)
    misfit2 = -0.5 * (sig2**-2) * np.sum((lambda2_s - data2) ** 2)

    LL = const_term + misfit1 + misfit2
    return LL


def test_2dof_tmcmc(tmp_path):
    """Test TMCMC on 2DOF example"""
    # Use temporary directory for status file
    status_file = tmp_path / "status_file_2DOF_case2_pba.txt"

    t = TMCMC(
        N=N,
        parameters=all_pars,
        names=["theta_1", "theta_2"],
        log_likelihood=log_likelihood_case2,
        status_file_name=str(status_file),
    )

    mytrace = t.run()

    assert mytrace is not None
