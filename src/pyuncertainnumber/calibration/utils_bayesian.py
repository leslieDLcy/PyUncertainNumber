import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, uniform
from scipy.stats import beta, norm, multivariate_normal, truncnorm

# utils_bayesian.py
import numpy as np
from typing import List

class Prior:
    """Abstract prior interface (minimal)."""
    def generate_rns(self, n: int) -> np.ndarray:
        """Return n samples ~ prior (shape: (n, d_i) or (n,))."""
        raise NotImplementedError
    def log_pdf_eval(self, x: np.ndarray) -> float:
        """Return log p(x) for a single vector x."""
        raise NotImplementedError

class PriorUniform(Prior):
    """Axis-aligned uniform prior over [lb, ub] in R^d (vector bounds)."""
    def __init__(self, lb: np.ndarray, ub: np.ndarray, name: str = "uniform"):
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        assert self.lb.shape == self.ub.shape
        self.d = self.lb.size
        self.name = name
        self._logZ = -np.sum(np.log(self.ub - self.lb))

    def generate_rns(self, n: int) -> np.ndarray:
        """Vectorized sampling over the hyper-rectangle."""
        return self.lb + (self.ub - self.lb) * np.random.rand(n, self.d)

    def log_pdf_eval(self, x: np.ndarray) -> float:
        """log 1/Vol if x within bounds; -inf otherwise."""
        x = np.asarray(x, float).ravel()
        if np.all(x >= self.lb) and np.all(x <= self.ub):
            return self._logZ
        return -np.inf

def initial_population(n: int, priors: List[Prior]) -> np.ndarray:
    """
    """  # compact: draw n particles across concatenated priors
    blocks = [p.generate_rns(n) for p in priors]
    # Each block may be (n,) or (n,d_i); ensure 2D then hstack
    blocks = [b.reshape(n, -1) for b in blocks]
    return np.hstack(blocks)

def log_prior(x: np.ndarray, priors: List[Prior]) -> float:
    """
    """  # compact: sum component prior logs for a single particle
    x = np.asarray(x, float).ravel()
    logs = []
    off = 0
    for p in priors:
        d = getattr(p, "d", 1)
        logs.append(p.log_pdf_eval(x[off:off+d]))
        off += d
    return float(np.sum(logs))

def log_prior_batch(samples: np.ndarray, priors: List[Prior]) -> np.ndarray:
    """
    """  # compact: batch version of log_prior
    samples = np.atleast_2d(samples)
    n, D = samples.shape
    out = np.zeros(n, float)
    off = 0
    for p in priors:
        d = getattr(p, "d", 1)
        # evaluate each row on its slice
        sl = samples[:, off:off+d]
        out += np.array([p.log_pdf_eval(sl[i]) for i in range(n)], float)
        off += d
    return out

def compute_beta_update_evidence(beta: float,
                                 log_lik: np.ndarray,
                                 log_evidence: float,
                                 prev_ESS: float,
                                 target_frac: float = 0.95,
                                 min_particles: int = 50,
                                 max_iter: int = 30) -> tuple:
    """
    """  # compact: binary-search next beta to hit target ESS ~ target_frac * prev_ESS
    old_beta = beta
    low, high = beta, 1.0
    target = max(target_frac * prev_ESS, min_particles)
    N = log_lik.size
    # numeric stabilize
    ll_max = float(log_lik.max())
    new_beta, Wn, ESS = beta, np.ones(N)/N, prev_ESS
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        inc = mid - old_beta
        W = np.exp(inc * (log_lik - ll_max))
        Wn = W / (W.sum() + 1e-300)
        ESS = 1.0 / np.sum(Wn**2)
        if abs(ESS - target) < 1.0:
            new_beta = mid
            break
        if ESS < target:
            high = mid
        else:
            low = mid
        new_beta = mid
        if (high - low) < 1e-8:
            break
    if new_beta >= 1.0:  # clamp & recompute at 1.0
        new_beta = 1.0
        inc = new_beta - old_beta
        W = np.exp(inc * (log_lik - ll_max))
        Wn = W / (W.sum() + 1e-300)
        ESS = 1.0 / np.sum(Wn**2)
    # evidence update (log of mean unnormalized weights)
    log_evidence += np.log(W.mean() + 1e-300)
    return new_beta, log_evidence, Wn, ESS

def gaussian_proposals(current: np.ndarray, cov: np.ndarray, n: int) -> np.ndarray:
    """
    """  # compact: draw n proposals ~ N(current, cov)
    D = current.size
    return current + np.random.multivariate_normal(np.zeros(D), cov, size=n)



class JointBivariatePDF:
    """Base class for 2D aleatoric samplers with copulas."""
    def __init__(self, params):
        self.eps = 1e-9
        self.params = params

    def sample(self, ns: int = 1000):
        raise NotImplementedError("implement sampling here.")


class BetaGaussCopulaPDF(JointBivariatePDF):
    """
    Aleatoric sampler: (xa1, xa2) ~ Beta marginals + Gaussian copula.
    Params: [a1, a2, b1, b2, rho] with values in [0, 1].
    Scaled internally via factor `s` (default: 20).
    """
    def __init__(self, params, s=20):
        super().__init__(params)
        a1, a2, b1, b2, rho = params
        self.a1, self.a2 = a1 * s, a2 * s
        self.b1, self.b2 = b1 * s, b2 * s
        self.rho = rho * 2 - 1  # map [0,1] → [-1,1]
        self.cov = np.array([[1.0, self.rho], [self.rho, 1.0]])

    def sample(self, ns: int = 1000):
        z = multivariate_normal(mean=[0, 0], cov=self.cov).rvs(size=ns)
        u = np.clip(norm.cdf(z), self.eps, 1 - self.eps)
        xa1 = beta(self.a1, self.a2).ppf(u[:, 0])
        xa2 = beta(self.b1, self.b2).ppf(u[:, 1])
        return np.stack([xa1, xa2], axis=1)


class NormalGaussCopulaPDF(JointBivariatePDF):
    """ Aleatoric sampler: (xa1, xa2) ~ Truncated Normal marginals """
    def __init__(self, params, s_mu=2, s_sigma=2):
        super().__init__(params)
        mu1, sigma1, mu2, sigma2, rho = params
        # Scale parameters to their actual domain
        self.mu1, self.sigma1 = (mu1 *2 - 1) * s_mu, sigma1 * s_sigma
        self.mu2, self.sigma2 = (mu2 * 2 - 1) * s_mu, sigma2 * s_sigma
        self.rho = rho * 2 - 1  # map [0,1] → [-1,1]
        self.cov = np.array([[1.0, self.rho], [self.rho, 1.0]])

    def sample(self, ns: int = 1000):
        # Step 1: Draw from Gaussian copula
        z = multivariate_normal(mean=[0, 0], cov=self.cov).rvs(size=ns)
        u = np.clip(truncnorm.cdf(z, -5, 5), self.eps, 1 - self.eps)

        # Step 2: Truncated normal marginals on [0,1]
        a1, b1 = (0 - self.mu1) / self.sigma1, (1 - self.mu1) / self.sigma1
        a2, b2 = (0 - self.mu2) / self.sigma2, (1 - self.mu2) / self.sigma2

        x1 = truncnorm.ppf(u[:, 0], a=a1, b=b1, loc=self.mu1, scale=self.sigma1)
        x2 = truncnorm.ppf(u[:, 1], a=a2, b=b2, loc=self.mu2, scale=self.sigma2)

        return np.stack([x1, x2], axis=1)  # shape (ns, 2)


class BetaMixtureGaussCopulaPDF(JointBivariatePDF):
    """
    Mixture of Beta marginals with Gaussian copula correlation.
    Each component uses (a1, a2), (b1, b2) marginals + rho.
    Params: flat vector of length n_param * n_mix with values in [0, 1]; scaled internally.
    """

    def __init__(self, params, n_mix: int =2, s: int =20, weights=None):
        super().__init__(params)
        self.n_mix = n_mix
        self.s = s
        self.eps = 1e-6  # numerical stability

        expected_len = n_mix * 5
        assert len(params) == expected_len, f"Expected {expected_len} parameters for n_mix={n_mix}"

        self.components = []
        for i in range(n_mix):
            a1, a2, b1, b2, rho = params[5 * i: 5 * (i + 1)]
            comp = {
                'a1': a1 * s,
                'a2': a2 * s,
                'b1': b1 * s,
                'b2': b2 * s,
                'rho': rho * 2 - 1,  # map [0,1] → [-1,1]
            }
            comp['cov'] = np.array([[1.0, comp['rho']], [comp['rho'], 1.0]])
            self.components.append(comp)

        # Mixture weights (uniform if none provided)
        if weights is None:
            self.weights = np.ones(n_mix) / n_mix
        else:
            assert len(weights) == n_mix
            self.weights = np.array(weights) / np.sum(weights)

    def sample(self, ns: int = 1000):
        counts = np.random.multinomial(ns, self.weights)
        samples = []

        for k, count in enumerate(counts):
            if count <=1 :
                count = 2  # Ensure at least 2 samples per component for stability
            comp = self.components[k]
            z = multivariate_normal(mean=[0, 0], cov=comp['cov']).rvs(size=count)
            u = np.clip(norm.cdf(z), self.eps, 1 - self.eps)
            xa1 = beta(comp['a1'], comp['a2']).ppf(u[:, 0])
            xa2 = beta(comp['b1'], comp['b2']).ppf(u[:, 1])
            samples.append(np.stack([xa1, xa2], axis=1))

        return np.vstack(samples)[:ns, :]


def compute_beta_update_evidence(beta, log_likelihoods, log_evidence, prev_ESS):
    """  Computes beta for the next stage and updated model evidence
    Parameters
    ----------
        beta : float  stage parameter.
        log_likelihoods : numpy array of size N  log likelihood values at all particles
        log_evidence : float  log of evidence.
        prev_ESS : int   effective sample size of previous stage
    Returns
    -------
        new_beta : float    stage parameter for next stage.
        log_evidence : float  updated log evidence.
        Wm_n : numpy array of size N    weights of particles for the next stage
        ESS : float      effective sample size of new stage
    """
    old_beta, min_beta, max_beta = beta, beta, 2.0
    N, rN = len(log_likelihoods), max(0.95 * prev_ESS, 50)  # min particles 50

    while max_beta - min_beta > 1e-6:  # min step size
        new_beta = 0.5 * (max_beta + min_beta)
        inc_beta = new_beta - old_beta
        weights = np.exp(inc_beta * (log_likelihoods - log_likelihoods.max()))
        wm_normalized = weights / sum(weights)
        ess = int(1 / np.sum(wm_normalized ** 2))
        if ess == rN:
            break
        elif ess < rN:
            max_beta = new_beta
        else:
            min_beta = new_beta

    if new_beta >= 1:
        new_beta = 1
        inc_beta = new_beta - old_beta  # plausible weights of Sm corresponding to new beta
        weights = np.exp(inc_beta * (log_likelihoods - log_likelihoods.max()))
        wm_normalized = weights / sum(weights)
    log_evidence = log_evidence + np.log((sum(weights) / N))

    return new_beta, log_evidence, wm_normalized, ess


def get_log_prior_samples(samples, all_pars):
    """ Computes log_prior value at all particles
    s : numpy array of size N x Np (N samples x N parameters)
    all_pars:  list of PDFs objects, length Np
        all_pars[i]: is a PDF object with a callable log_pdf_eval method.
    Returns
     log_p : numpy array of size N log prior at all N particles .
    """
    log_p = 0
    for i in range(len(samples)):
        log_p = log_p + all_pars[i].log_pdf_eval(samples[i])
    return log_p


def initial_population(N, all_pars) -> np.ndarray:
    """
    Generates initial population from prior distribution
    N : float   number of particles.
    all_pars : list of size Np is number of parameters; all_pars[i] is object of type pdfs
    ini_pop : numpy array of size N x Np  initial population.
    """
    ini_pop = np.zeros((N, len(all_pars)))
    for i in range(len(all_pars)):
        ini_pop[:, i] = all_pars[i].generate_rns(N)
    return ini_pop


def prepare_all_pars(n_epistemic: int = 3, n_mix: int = 2):
    """
    Prepare prior for the NASA case study with a mixture of Beta marginals + copula correlation.
    Each mixture component gets its own (a1, a2, b1, b2, rho), scaled ∈ [0.001, 0.999].
    """
    l, u = 0.001, 0.999
    epistemic_domain = {}
    for ei in range(n_epistemic):
        epistemic_domain.update({f'xe{ei}': (l, u)})

    all_pars = []
    # Epistemic priors (uniform)
    for name, (lb, ub) in epistemic_domain.items():
        all_pars.append(PriorUniform(lb, ub, name=name))

    # Mixture components: (a1, a2, b1, b2, rho) × n_mix
    for i in range(n_mix):
        all_pars.append(PriorUniform(l, u, name=f'a1_m{i}'))
        all_pars.append(PriorUniform(l, u, name=f'a2_m{i}'))
        all_pars.append(PriorUniform(l, u, name=f'b1_m{i}'))
        all_pars.append(PriorUniform(l, u, name=f'b2_m{i}'))
        all_pars.append(PriorUniform(l, u, name=f'rho_m{i}'))

    return all_pars


def prepare_prior_beta_model_NASA():
    """ Prepare the prior for the beta model used in the NASA case study."""
    l, u = 0.001, 0.999
    epistemic_domain = {
        'xe1':  (l, u),
        'xe2':  (l, u),
        'xe3':  (l, u),
        'xa1a': (l, u), # marginal 1
        'xa2a': (l, u),
        'xa1b': (l, u), # marginal 2
        'xa2b': (l, u),
        'rho':  (l, u),  #  copula param
    }

    lb = np.array([v[0] for v in epistemic_domain.values()])
    ub = np.array([v[1] for v in epistemic_domain.values()])
    param_names = list(epistemic_domain.keys())
    all_pars = [PriorUniform(l, u, name=n) for l, u, n in zip(lb, ub, param_names)]
    return all_pars, epistemic_domain


def propose(current, covariance, n):
    """  proposal distribution for MCMC in perturbation stage"""
    return np.random.multivariate_normal(current, covariance, n)



def MCMC_MH(particle_num,
            Em,
            Nm_steps,
            current,
            likelihood_current,
            posterior_current,
            beta,
            numAccepts,
            all_pars,
            log_likelihood,
            ):
    """
    Markov chain Monte Carlo using Metropolis-Hastings
    "perturbs" each particle using MCMC-MH

    Parameters
    ----------
    particle_num : int
        particle number
    Em : numpy array of size Np x Np
        proposal covarince matrix.
    Nm_steps : int
        number of perturbation steps.
    current : numpy array of size Np
        current particle location
    likelihood_current : float
        log likelihood value at current particle
    posterior_current : float
        log posterior value at current particle
    beta : float
        stage parameter.
    numAccepts : int
        total number of accepts
    all_pars : : list of size Np
        Np is number of parameters
        all_pars[i] is object of type pdfs
        all parameters to be inferred.
    log_likelihood : function
        log likelihood function to be defined in main.py.

    Returns
    -------
    current : numpy array of size Np
        perturbed particle location
    likelihood_current : float
        log likelihood value at perturbed particle
    posterior_current : float
        log posterior value at perturbed particle
    numAccepts : int
        total number of accepts during perturbation (MCMC - MH)

    """
    all_proposals, all_PLP = [],  []
    deltas = propose(np.zeros(len(current)), Em, Nm_steps)
    print(f"MCMC_MH: running particle = {particle_num}, for steps {Nm_steps}" )

    for j2 in range(Nm_steps):
        delta = deltas[j2]
        proposal = current + delta
        prior_proposal = log_prior(proposal, all_pars)

        if not np.isfinite(prior_proposal):

            likelihood_proposal = -np.inf  # dont run the model
            posterior_proposal = -np.inf
        # proposal satisfies the prior constraints
        else:
            likelihood_proposal = log_likelihood(particle_num, proposal)
            posterior_proposal = prior_proposal + likelihood_proposal * beta

        log_acceptance = posterior_proposal - posterior_current

        all_proposals.append(proposal)
        all_PLP.append([prior_proposal, likelihood_proposal, posterior_proposal])

        if np.isfinite(log_acceptance) and (np.log(np.random.uniform()) < log_acceptance):
            current = proposal  # accept
            posterior_current = posterior_proposal
            likelihood_current = likelihood_proposal
            numAccepts += 1

    # gather all last samples
    return current, likelihood_current, posterior_current, numAccepts
