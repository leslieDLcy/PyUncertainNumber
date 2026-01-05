from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from numpy.typing import NDArray
import pickle
from scipy.stats import wasserstein_distance, entropy
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import time
import numpy as np
import logging
from ..console import console

if TYPE_CHECKING:
    from pyuncertainnumber import Distribution
    from .pdfs import ProbabilityDensityFun

"""
This is the implementation for Transitional Markov Chain Monte Carlo (TMCMC) algorithm

Leslie refactored, revised and integrated for `pyuncertainnumber` package, based on the version from 
Roberto Rocchetta (NASA UQ challenge 2025). The original code is from Mukesh K. Ramancha.
@license: MIT License
@date: Nov 2025
"""


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class TMCMC:
    """Class for TMCMC implementation

    args:
        N (int) : number of particles to be sampled from posterior

        parameters (list) : list of (size Nop) prior distributions instances

        log_likelihood (callable): log likelihood function on :math:`\\theta` to be defined which is problem specific

        status_file_name (str): name of the status file to store status of the tmcmc sampling

    return:
        mytrace: returns trace file of all samples of all tmcmc stages.
            At stage m: it contains [Sm, Lm, Wm_n, ESS, beta, Smcap]


    example:
        >>> from pyuncertainnumber.calibration.tmcmc import TMCMC
        >>> from pyuncertainnumber import pba
        >>> # create an instance of TMCMC class and run tmcmc
        >>> parameters = [pba.D("uniform", (0.8, 2.2)), pba.D("uniform", (0.4, 1.2))] # dummy prior distributions

        >>> t = TMCMC(
        >>>     N=1000,                                             # number of particles
        >>>     parameters=parameters,                              # prior distributions
        >>>     names =['k1', 'k2'],                                # parameter names
        >>>     log_likelihood=log_likelihood_function,             # user defined log likelihood function
        >>>     mutation_steps=1,                                   # number of MCMC steps for perturbation
        >>>     status_file_name='tmcmc_progressing_status.txt',    # status log file
        >>> )
        >>> mytrace = t.run()

        >>> # for loading trace file
        >>> import pickle
        >>> with open('tmcmc_trace.pkl', 'rb') as f:
        >>>     mytrace = pickle.load(f)
        >>> re = TMCMC(trace=mytrace, names=names)                  # create TMCMC instance with loaded trace and names

        >>> re.plot_updated_distribution(save=False)                # plot prior and posterior distribution
        >>> prior_samples = re.load_prior_samples()                 # load prior samples as DataFrame
        >>> posterior_samples = re.load_posterior_samples()         # load posterior samples as DataFrame


    .. figure:: /_static/tmcmc_2dof.png
        :alt: 2DOF TMCMC example
        :align: center
        :width: 70%

        Example of TMCMC calibration of a 2-DOF system
    """

    def __init__(
        self,
        N: int = None,
        parameters: list = None,
        names: list[str] = None,
        log_likelihood: callable = None,
        mutation_steps: int = 5,
        status_file_name: str = None,
        trace=None,
    ):
        self.N = N
        self.parameters = parameters
        self.names = names
        self.log_likelihood = log_likelihood
        self.mutation_steps = mutation_steps
        self.status_file_name = status_file_name
        self.trace = trace
        self.check_names()

    def check_names(self):
        if self.names is None:
            raise ValueError("Parameter names must be  provided.")

    def run(self) -> list[Stage]:
        """Run the TMCMC algorithm

        returns:
            mytrace: returns trace file of all samples of all tmcmc stages.
        """

        mytrace = run_tmcmc_updated(
            self.N,
            all_pars=self.parameters,
            log_likelihood=self.log_likelihood,
            Nm_steps_max=self.mutation_steps,
            status_file_name=self.status_file_name,
        )
        self.trace = mytrace
        self.check_trace()
        return mytrace

    def check_trace(self):
        if hasattr(self, "trace") and self.trace is not None:
            print("Trace is loaded.")

    def save_trace(self, mytrace, file_name: str, save_dir=None):
        """Save the trace to a file

        args:
            mytrace: trace file of all samples of all tmcmc stages.
            file_name (str): name of the file to save the trace
        """
        # save results
        if save_dir is not None:
            file_name = f"{save_dir}/{file_name}"

        with open(f"{file_name}.pkl", "wb") as f:
            pickle.dump(mytrace, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_prior_samples(self, trace=None) -> pd.DataFrame:
        if trace is None:
            trace = self.trace
        prior = pd.DataFrame(trace[0].samples, columns=self.names)  # samples from prior
        return prior

    def load_posterior_samples(self, trace=None) -> pd.DataFrame:
        if trace is None:
            trace = self.trace
        posterior = pd.DataFrame(trace[-1].samples, columns=self.names)
        return posterior

    def plot_updated_distribution(self, trace=None, save=False):
        """Plot the prior and posterior distribution of the parameters

        args:
            trace: trace file of all samples of all tmcmc stages.
            save (bool): if True, save the figure
        """
        if trace is None:
            trace = self.trace

        if not hasattr(self, "names") or self.names is None:
            raise ValueError("Parameter names are not provided.")

        plot_updated_distribution(trace, self.names, save=save)


@dataclass
class Stage:
    """Container for one TMCMC stage.

    Args:
        Sm (NDArray): samples before resampling (N x Np)
        Lm (NDArray): log-likelihoods at Sm (N,)
        Wm_n (NDArray): normalized incremental weights (N,)
        ESS (Optional[float]): effective sample size the next stage (can be None in terminal stage)
        beta (float): tempering parameter next stage beta
        Smcap (Optional[NDArray]): resampled samples (N x Np), None in terminal stage
    """

    Sm: NDArray  # samples before resampling (N x Np)
    Lm: NDArray  # log-likelihoods at Sm (N,)
    Wm_n: NDArray  # normalized incremental weights (N,)
    ESS: Optional[
        float
    ]  # effective sample size next stage ESS (can be None in terminal stage)
    beta: float  # tempering parameter next stage beta
    Smcap: Optional[NDArray]  # resampled samples (N x Np), None in terminal stage

    def __repr__(self) -> str:
        return (
            f"Stage(beta={self.beta:.4f}, " f"ESS={self.ESS}, " f"N={self.Sm.shape[0]})"
        )

    # * ---- Hints ----

    @property
    def samples(self) -> NDArray:
        """Samples before resampling (from previous stage)."""
        return self.Sm

    @property
    def log_likelihoods(self) -> NDArray:
        """Log-likelihood evaluated at `samples`."""
        return self.Lm

    @property
    def weights(self) -> NDArray:
        """Normalized incremental importance weights."""
        return self.Wm_n

    @property
    def effective_sample_size(self) -> Optional[float]:
        """Effective sample size after reweighting the next stage."""
        return self.ESS

    @property
    def tempering_parameter(self) -> float:
        """Tempering parameter beta next stage beta."""
        return self.beta

    @property
    def resampled_samples(self) -> Optional[NDArray]:
        """Samples after importance resampling (before MH mutation)."""
        return self.Smcap


def recover_trace_samples(mytrace: list[Stage], names: list[str]) -> pd.DataFrame:
    """Load back the Prior and Posterior samples in the form of a DataFrame

    args:
        mytrace (list[Stage]): trace file of all samples of all tmcmc stages.
        names (list[str]): list of parameter names

    returns:
        (pd.DataFrame) : Prior and Posterior in the trace as a DataFrame
    """

    df1 = pd.DataFrame(mytrace[0].samples, columns=names)  # samples from prior
    df2 = pd.DataFrame(
        mytrace[-1].samples, columns=names
    )  # samples from last step posterior

    # Add a column to identify the source
    df1["source"] = "Prior Samples"
    df2["source"] = "Posterior Samples"

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2])

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2])
    return combined_df


# * ----------------------------------- tmcmc


def plot_updated_distribution(mytrace, names: list[str], save=False):
    """Plot the prior and posterior distribution of the parameters

    args:
        mytrace: trace file of all samples of all tmcmc stages.
        names (list[str]): list of parameter names
        save (bool): if True, save the figure
    """

    # Example DataFrames from mytrace
    # names = list(Config.epistemic_domain.keys())

    # names = ["Xe_1", "Xe_2", "Xe_3", "Xa1_mu", "Xa2_mu", "Xa1_std", "Xa2_std"]
    df1 = pd.DataFrame(mytrace[0].samples, columns=names)  # samples from prior
    df2 = pd.DataFrame(
        mytrace[-1].samples, columns=names
    )  # samples from last step posterior

    # Add a column to identify the source
    df1["source"] = "Prior Samples"
    df2["source"] = "Posterior Samples"

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2])

    # Combine the DataFrames
    combined_df = pd.concat([df1, df2])

    # Create the pairplot with density plots on the lower triangle
    g = sb.pairplot(
        combined_df,
        hue="source",
        palette="Set2",
        diag_kind="kde",  # Add KDE plots to the diagonal
    )

    # Add density plots on the lower triangle
    for i, j in zip(
        *np.tril_indices_from(g.axes, -1)
    ):  # Iterate over the lower triangle
        sb.kdeplot(
            data=combined_df,
            x=combined_df.columns[j],
            y=combined_df.columns[i],
            hue="source",
            fill=True,
            alpha=0.5,
            ax=g.axes[i, j],
        )
    if save == True:
        print("Note: figure saved")
        plt.savefig("updated_distribution.png")
    plt.show()


def initial_population(N: float, all_pars: list) -> NDArray:
    """Generates initial population from prior distribution

    Args:
        N (float): number of particles.

        all_pars (list) : All the parameters, of size Np, to be updated. `all_pars[i]` is object of type `pdfs`.

    Returns
        ini_pop (NDArray): the initial population which is a numpy array of size `N x Np`.

    note:
        `Np` is number of parameters
    """
    ini_pop = np.zeros((N, len(all_pars)))
    for i in range(len(all_pars)):
        ini_pop[:, i] = all_pars[i].generate_rns(N)
    return ini_pop


def log_prior(
    s: NDArray, all_pars: list[ProbabilityDensityFun | Distribution]
) -> float:
    """Computes log_prior value at all particles

    Args:
        s (NDArray): numpy array of size (N x Np) of all particles.

        all_pars (list[ProbabilityDensityFun | Distribution]) : All the parameters, of size Np, to be updated. `all_pars[i]` is object of type `pdfs`.

    Returns
        log_p (NDArray): log prior at all N particles which is a numpy array of size N

    Note:
        `Np` denotes number of parameters
    """
    log_p = 0
    for i in range(len(s)):
        log_p = log_p + all_pars[i].log_pdf_eval(s[i])
    return log_p


def compute_beta_update_evidence(
    beta: float, log_likelihoods: NDArray, log_evidence: float, prev_ESS: int
):
    """Computes beta for the next stage and updated model evidence

    Args:
        beta (float): stage parameter.
        log_likelihoods (NDArray): log likelihood values at all particles which is numpy array of size N
        log_evidence (float): log of evidence.
        prev_ESS (int): effective sample size of previous stage

    :returns: A 4-tuple ``(new_beta, log_evidence, Wm_n, ESS)`` where:
        - **new_beta** (*float*): stage parameter for next stage.
        - **log_evidence** (*float*): updated log evidence.
        - **Wm_n** (*NDArray*): weights of particles for the next stage.
        - **ESS** (*float*): effective sample size of new stage.
    :rtype: tuple[float, float, NDArray, float]

    """
    old_beta = beta
    min_beta = beta
    max_beta = 2.0
    N = len(log_likelihoods)
    # rN = int(len(log_likelihoods) * 0.5)
    # rN = 0.95*prev_ESS
    rN = max(0.95 * prev_ESS, 50)  # min particles 50

    while max_beta - min_beta > 1e-8:  # min step size
        new_beta = 0.5 * (max_beta + min_beta)

        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta - old_beta

        Wm = np.exp(inc_beta * (log_likelihoods - log_likelihoods.max()))
        Wm_n = Wm / sum(Wm)
        ESS = int(1 / np.sum(Wm_n**2))

        # log_Wm = inc_beta * log_likelihoods
        # log_Wm_n = log_Wm - logsumexp(log_Wm)
        # ESS = int(np.exp(-logsumexp(log_Wm_n * 2)))

        if ESS == rN:
            break
        elif ESS < rN:
            max_beta = new_beta
        else:
            min_beta = new_beta

    if new_beta >= 1:
        new_beta = 1

        # plausible weights of Sm corresponding to new beta
        inc_beta = new_beta - old_beta

        Wm = np.exp(inc_beta * (log_likelihoods - log_likelihoods.max()))
        Wm_n = Wm / sum(Wm)

        # log_Wm = inc_beta * log_likelihoods
        # log_Wm_n = log_Wm - logsumexp(log_Wm)

    # Wm = np.exp(log_Wm)
    # Wm_n = np.exp(log_Wm_n)

    # update model evidence
    # (check it, might not be correct, as we remove log.likelihood max in compute_beta)
    # evidence = evidence * (sum(Wm)/N)
    # log_evidence = log_evidence + logsumexp(log_Wm) - np.log(N)
    log_evidence = log_evidence + np.log((sum(Wm) / N))

    return new_beta, log_evidence, Wm_n, ESS


def propose(current: NDArray, covariance: NDArray, n: int) -> NDArray:
    """Proposal distribution for MCMC in pertubation stage

    Args:
        current (NDArray): numpy array of size Np
            current particle location
        covariance (NDArray): numpy array of size Np x Np
            proposal covariance matrix
        n (int): number of proposals.

    Returns:
        NDArray: n proposals
    """
    return np.random.multivariate_normal(current, covariance, n)


def MCMC_MH(
    particle_num: int,
    Em: NDArray,
    Nm_steps: int,
    current: NDArray,
    likelihood_current: float,
    posterior_current: float,
    beta: float,
    numAccepts: int,
    all_pars: list,
    log_likelihood: callable,
) -> tuple[NDArray, float, float, int]:
    """Markov chain Monte Carlo using Metropolis-Hastings which "perturbs" each particle using MCMC-MH

    Conduct `Nm_steps` steps of MH for each particle.

    Args:
        particle_num (int) : The index of the current particle/parameter vector being evaluated.

        Em (NDArray) : proposal covarince matrix which is numpy array of size Nop x Nop

        Nm_steps (int) : number of perturbation steps.

        current (NDArray) : numpy array of size Nop. current particle location

        likelihood_current (float) : log likelihood value at current particle

        posterior_current (float) : log posterior value at current particle

        beta (float) : stage parameter.

        numAccepts (int) : total number of accepts

        all_pars : list of size Nop
            Nop is number of parameters
            all_pars[i] is object of type pdfs
            all parameters to be inferred.

        log_likelihood (callable): log likelihood function to be defined in main.py.

    :returns: A 4-tuple ``(current, likelihood_current, posterior_current, numAccepts)`` where:

        - **current** (*NDArray*): Perturbed particle location of shape (Nop,).
        - **likelihood_current** (*float*): Log-likelihood at the perturbed particle.
        - **posterior_current** (*float*): Log-posterior at the perturbed particle.
        - **numAccepts** (*int*): Updated total number of accepts.
    :rtype: tuple[NDArray, float, float, int]

    """
    all_proposals = []
    all_PLP = []

    deltas = propose(np.zeros(len(current)), Em, Nm_steps)
    # logging.info(
    #     f"MCMC_MH: running for particle_num = {particle_num}, .... N chains {Nm_steps}"
    # )

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

        if np.isfinite(log_acceptance) and (
            np.log(np.random.uniform()) < log_acceptance
        ):
            # accept
            current = proposal
            posterior_current = posterior_proposal
            likelihood_current = likelihood_proposal
            numAccepts += 1

    # gather all last samples
    return current, likelihood_current, posterior_current, numAccepts


# TODO
def run_tmcmc_vec():
    """vectorised version of run_tmcmc to be implemented later"""
    pass


def run_tmcmc_updated(
    N: int,
    all_pars: list,
    log_likelihood: callable,
    status_file_name: str,
    Nm_steps_max: int = 5,
    Nm_steps_maxmax: int = 5,
) -> list[Stage]:
    """Updated main workflow of running Transitional MCMC

    Solves the "At this point, beta has been updated to the new value, but Postm is still the log-target from the previous beta (except at the very first stage).
    So Postmcap is inconsistent with the beta you pass into MCMC_MH."

    args:
        N  (int) : int
            number of particles to be sampled from posterior

        all_pars (list) : All the parameters, of size Nop, to be updated. `all_pars[i]` is object of type `pdfs`.

        log_likelihood (callable): log likelihood function to be defined in main.py as is problem specific

        status_file_name (str): name of the status file to store status of the tmcmc sampling

        Nm_steps_max (int, optional): Numbers of MCMC steps for perturbation. The default is 5.

        Nm_steps_maxmax (int, optional): Numbers of MCMC steps for perturbation. The default is 5.

    returns:
        list[Stage]: the trace that contains all iterations of the Stage instances.
    """
    parallel_processing = "multiprocessing"

    # side note: make all_pars as ordered dict in the future
    # Initialize (beta, effective sample size)
    beta = 0
    ESS = N
    mytrace = []
    stage_num = 0
    start_time_global = time.time()

    # Initialize other TMCMC variables
    Nm_steps = Nm_steps_max
    parallelize_MCMC = True
    Adap_calc_Nsteps, Adap_scale_cov = "yes", "yes"  # yes or no
    scalem = 1  # cov scale factor
    log_evidence = 0  # model evidence

    # initial samples -> array (N, Np)
    Sm = initial_population(N, all_pars)

    # Evaluate posterior at Sm
    Priorm = np.array([log_prior(s, all_pars) for s in Sm]).squeeze()
    Postm = Priorm  # prior = post for beta = 0

    status_file = open(status_file_name, "a+")

    # Evaluate log-likelihood at current samples Sm
    if parallelize_MCMC:
        iterables = [(ind, Sm[ind]) for ind in range(N)]
        status_file.write("======================== \n")
        if parallel_processing == "multiprocessing":
            status_file.write("using multiprocessing \n")
            import multiprocessing as mp
            from multiprocessing import Pool

            pool = Pool(processes=mp.cpu_count() - 2)
            Lmt = pool.starmap(log_likelihood, iterables)

        else:
            raise (
                AssertionError(
                    "parallel_processing invalid, should be either multiprocessing or mpi"
                )
            )
        status_file.write("======================== \n")
        Lm = np.array(Lmt).squeeze()
    else:
        Lm = []
        for ind in range(N):
            Lm.append(log_likelihood(ind, Sm[ind]))
            logging.info(
                f"Computing likelihood stage {stage_num}: particle:{ind + 1}/{N}"
            )
        Lm = np.array(Lm).squeeze()
    status_file.close()

    while beta < 1:
        stage_num += 1
        start_time_stage = time.time()

        # adaptivly compute beta s.t. ESS = N/2 or ESS = 0.95*prev_ESS
        # plausible weights of Sm corresponding to new beta
        # logging.info(f"'Computing beta the weights ...")

        beta_old = beta
        beta, log_evidence, Wm_n, ESS = compute_beta_update_evidence(
            beta, Lm, log_evidence, ESS
        )

        # NEW: update Postm to match the NEW beta, before resampling / MH
        Postm = Postm + (beta - beta_old) * Lm

        console.log(
            f"[bold green]TMCMC Iteration stage {stage_num}: Tempering parameter updated to {beta:.6f}[/bold green]"
        )

        # Calculate covaraince matrix using Wm_n
        Cm = np.cov(Sm, aweights=Wm_n, rowvar=0)

        # * --------------------------------------------------------- Resample
        # Resampling using plausible weights
        SmcapIDs = np.random.choice(range(N), N, p=Wm_n)
        # SmcapIDs = resampling.stratified_resample(Wm_n)
        Smcap = Sm[SmcapIDs]  # Resampled particles
        Lmcap = Lm[SmcapIDs]
        Postmcap = Postm[SmcapIDs]

        #  save to trace
        # stage m: samples, likelihood, weights, next stage ESS, next stage beta, resampled samples
        mytrace.append(Stage(Sm=Sm, Lm=Lm, Wm_n=Wm_n, ESS=ESS, beta=beta, Smcap=Smcap))

        # TODO: plot updated distribution to fix later on
        # if stage_num in [2, 4, 6]:
        #     plot_updated_distribution(mytrace)

        # print to status_file
        status_file = open(status_file_name, "a+")
        status_file.write("stage number = %d \n" % stage_num)
        status_file.write("beta = %.5f \n" % beta)
        status_file.write("ESS = %d \n" % ESS)
        status_file.write("scalem = %.2f \n" % scalem)

        # * --------------------------------------------------------- MH Perturb
        # perform MCMC starting at each Smcap (total: N) for Nm_steps
        Em = ((scalem) ** 2) * Cm  # Proposal dist covariance matrix

        numProposals = N * Nm_steps
        numAccepts = 0

        if parallelize_MCMC:
            iterables = [
                (
                    j1,
                    Em,
                    Nm_steps,
                    Smcap[j1],
                    Lmcap[j1],
                    Postmcap[j1],
                    beta,
                    numAccepts,
                    all_pars,
                    log_likelihood,
                )
                for j1 in range(N)
            ]

            if parallel_processing == "multiprocessing":
                results = pool.starmap(MCMC_MH, iterables)

            # elif parallel_processing == "mpi":
            #     results = list(executor.starmap(MCMC_MH, iterables))
        else:
            """Here we are running Markov-Chain Monte Carlo for each particle"""
            results = [
                MCMC_MH(
                    j1,
                    Em,
                    Nm_steps,
                    Smcap[j1],
                    Lmcap[j1],
                    Postmcap[j1],
                    beta,
                    numAccepts,
                    all_pars,
                    log_likelihood,
                )
                for j1 in range(N)
            ]

        Sm1, Lm1, Postm1, numAcceptsS = zip(*results)
        Sm1 = np.asarray(Sm1)
        Lm1 = np.asarray(Lm1)
        Postm1 = np.asarray(Postm1)
        numAcceptsS = np.asarray(numAcceptsS)
        numAccepts = sum(numAcceptsS)

        # total observed acceptance rate
        R = numAccepts / numProposals
        status_file.write("acceptance rate = %.2f \n" % R)

        # Calculate Nm_steps based on observed acceptance rate
        if Adap_calc_Nsteps == "yes":
            # increase max Nmcmc with stage number
            Nm_steps_max = min(Nm_steps_max + 1, Nm_steps_maxmax)
            status_file.write("adapted max MCMC steps = %d \n" % Nm_steps_max)

            acc_rate = max(1.0 / numProposals, R)
            Nm_steps = min(
                Nm_steps_max, 1 + int(np.log(1 - 0.99) / np.log(1 - acc_rate))
            )
            status_file.write("next MCMC Nsteps = %d \n" % Nm_steps)

        status_file.write("log_evidence till now = %.20f \n" % log_evidence)
        status_file.write(
            "--- Execution time: %.2f mins --- \n"
            % ((time.time() - start_time_stage) / 60)
        )
        status_file.write("======================== \n")
        status_file.close()

        # scale factor based on observed acceptance ratio
        if Adap_scale_cov == "yes":
            scalem = (1 / 9) + ((8 / 9) * R)

        # for next beta
        Sm, Postm, Lm = Sm1, Postm1, Lm1

    # save to trace, which is the last stopping stage
    mytrace.append(
        Stage(
            Sm=Sm,
            Lm=Lm,
            Wm_n=np.ones(len(Wm_n)) / len(Wm_n),
            ESS=None,
            beta=1.0,
            Smcap=None,
        )
    )

    status_file = open(status_file_name, "a+")
    status_file.write(
        "--- Execution time: %.2f mins --- \n"
        % ((time.time() - start_time_global) / 60)
    )
    status_file.write("log_evidence = %.20f \n" % log_evidence)

    if parallelize_MCMC:
        if parallel_processing == "multiprocessing":
            status_file.write("closing multiprocessing \n")
            pool.close()

    status_file.close()

    if parallel_processing == "multiprocessing":
        return mytrace


def run_tmcmc(
    N: int,
    all_pars: list,
    log_likelihood: callable,
    status_file_name: str,
    Nm_steps_max: int = 5,
    Nm_steps_maxmax: int = 5,
):
    """Main workflow of running Transitional MCMC

    args:
        N  (int) : int
            number of particles to be sampled from posterior

        all_pars (list) : All the parameters, of size Nop, to be updated. `all_pars[i]` is object of type `pdfs`.

        log_likelihood (callable): log likelihood function to be defined in main.py as is problem specific

        status_file_name (str): name of the status file to store status of the tmcmc sampling

        Nm_steps_max (int, optional): Numbers of MCMC steps for perturbation. The default is 5.

        Nm_steps_maxmax (int, optional): Numbers of MCMC steps for perturbation. The default is 5.

        parallel_processing (str): should be either 'multiprocessing' or 'mpi'

    returns:
        a tuple containing:
            mytrace: returns trace file of all samples of all tmcmc stages.
                at stage m: it contains [Sm, Lm, Wm_n, ESS, beta, Smcap]
    """
    parallel_processing = "multiprocessing"

    # side note: make all_pars as ordered dict in the future
    # Initialize (beta, effective sample size)
    beta = 0
    ESS = N
    mytrace = []
    stage_num = 0
    start_time_global = time.time()

    # Initialize other TMCMC variables
    Nm_steps = Nm_steps_max
    parallelize_MCMC = True
    Adap_calc_Nsteps, Adap_scale_cov = "yes", "yes"  # yes or no
    scalem = 1  # cov scale factor
    log_evidence = 0  # model evidence

    # initial samples -> array (N, Np)
    Sm = initial_population(N, all_pars)

    # Evaluate posterior at Sm
    Priorm = np.array([log_prior(s, all_pars) for s in Sm]).squeeze()
    Postm = Priorm  # prior = post for beta = 0

    status_file = open(status_file_name, "a+")

    # Evaluate log-likelihood at current samples Sm
    if parallelize_MCMC:
        iterables = [(ind, Sm[ind]) for ind in range(N)]
        status_file.write("======================== \n")
        if parallel_processing == "multiprocessing":
            status_file.write("using multiprocessing \n")
            import multiprocessing as mp
            from multiprocessing import Pool

            pool = Pool(processes=mp.cpu_count() - 2)
            Lmt = pool.starmap(log_likelihood, iterables)

        else:
            raise (
                AssertionError(
                    "parallel_processing invalid, should be either multiprocessing or mpi"
                )
            )
        status_file.write("======================== \n")
        Lm = np.array(Lmt).squeeze()
    else:
        Lm = []
        for ind in range(N):
            Lm.append(log_likelihood(ind, Sm[ind]))
            logging.info(
                f"Computing likelihood stage {stage_num}: particle:{ind + 1}/{N}"
            )
        Lm = np.array(Lm).squeeze()
    status_file.close()

    while beta < 1:
        stage_num += 1
        start_time_stage = time.time()

        # adaptivly compute beta s.t. ESS = N/2 or ESS = 0.95*prev_ESS
        # plausible weights of Sm corresponding to new beta
        # logging.info(f"'Computing beta the weights ...")

        beta, log_evidence, Wm_n, ESS = compute_beta_update_evidence(
            beta, Lm, log_evidence, ESS
        )

        console.log(
            f"[bold green]TMCMC Iteration stage {stage_num}: Tempering parameter updated to {beta:.6f}[/bold green]"
        )

        # Calculate covaraince matrix using Wm_n
        Cm = np.cov(Sm, aweights=Wm_n, rowvar=0)

        # * --------------------------------------------------------- Resample
        # Resampling using plausible weights
        SmcapIDs = np.random.choice(range(N), N, p=Wm_n)
        # SmcapIDs = resampling.stratified_resample(Wm_n)
        Smcap = Sm[SmcapIDs]  # Resampled particles
        Lmcap = Lm[SmcapIDs]
        Postmcap = Postm[SmcapIDs]

        #  save to trace
        # stage m: samples, likelihood, weights, next stage ESS, next stage beta, resampled samples
        mytrace.append([Sm, Lm, Wm_n, ESS, beta, Smcap])

        # TODO: plot updated distribution to fix later on
        # if stage_num in [2, 4, 6]:
        #     plot_updated_distribution(mytrace)

        # print to status_file
        status_file = open(status_file_name, "a+")
        status_file.write("stage number = %d \n" % stage_num)
        status_file.write("beta = %.5f \n" % beta)
        status_file.write("ESS = %d \n" % ESS)
        status_file.write("scalem = %.2f \n" % scalem)

        # * --------------------------------------------------------- MH Perturb
        # perform MCMC starting at each Smcap (total: N) for Nm_steps
        Em = ((scalem) ** 2) * Cm  # Proposal dist covariance matrix

        numProposals = N * Nm_steps
        numAccepts = 0

        if parallelize_MCMC:
            iterables = [
                (
                    j1,
                    Em,
                    Nm_steps,
                    Smcap[j1],
                    Lmcap[j1],
                    Postmcap[j1],
                    beta,
                    numAccepts,
                    all_pars,
                    log_likelihood,
                )
                for j1 in range(N)
            ]

            if parallel_processing == "multiprocessing":
                results = pool.starmap(MCMC_MH, iterables)

            # elif parallel_processing == "mpi":
            #     results = list(executor.starmap(MCMC_MH, iterables))
        else:
            """Here we are running Markov-Chain Monte Carlo for each particle"""
            results = [
                MCMC_MH(
                    j1,
                    Em,
                    Nm_steps,
                    Smcap[j1],
                    Lmcap[j1],
                    Postmcap[j1],
                    beta,
                    numAccepts,
                    all_pars,
                    log_likelihood,
                )
                for j1 in range(N)
            ]

        Sm1, Lm1, Postm1, numAcceptsS = zip(*results)
        Sm1 = np.asarray(Sm1)
        Lm1 = np.asarray(Lm1)
        Postm1 = np.asarray(Postm1)
        numAcceptsS = np.asarray(numAcceptsS)
        numAccepts = sum(numAcceptsS)

        # total observed acceptance rate
        R = numAccepts / numProposals
        status_file.write("acceptance rate = %.2f \n" % R)

        # Calculate Nm_steps based on observed acceptance rate
        if Adap_calc_Nsteps == "yes":
            # increase max Nmcmc with stage number
            Nm_steps_max = min(Nm_steps_max + 1, Nm_steps_maxmax)
            status_file.write("adapted max MCMC steps = %d \n" % Nm_steps_max)

            acc_rate = max(1.0 / numProposals, R)
            Nm_steps = min(
                Nm_steps_max, 1 + int(np.log(1 - 0.99) / np.log(1 - acc_rate))
            )
            status_file.write("next MCMC Nsteps = %d \n" % Nm_steps)

        status_file.write("log_evidence till now = %.20f \n" % log_evidence)
        status_file.write(
            "--- Execution time: %.2f mins --- \n"
            % ((time.time() - start_time_stage) / 60)
        )
        status_file.write("======================== \n")
        status_file.close()

        # scale factor based on observed acceptance ratio
        if Adap_scale_cov == "yes":
            scalem = (1 / 9) + ((8 / 9) * R)

        # for next beta
        Sm, Postm, Lm = Sm1, Postm1, Lm1

    # save to trace, which is the last stopping stage
    mytrace.append([Sm, Lm, np.ones(len(Wm_n)) / len(Wm_n), "notValid", 1, "notValid"])

    status_file = open(status_file_name, "a+")
    status_file.write(
        "--- Execution time: %.2f mins --- \n"
        % ((time.time() - start_time_global) / 60)
    )
    status_file.write("log_evidence = %.20f \n" % log_evidence)

    if parallelize_MCMC:
        if parallel_processing == "multiprocessing":
            status_file.write("closing multiprocessing \n")
            pool.close()

    status_file.close()

    if parallel_processing == "multiprocessing":
        return mytrace


# * ----------------------------------- likelihood


def log_likelihood(
    sam_id: int,
    param_vec: NDArray,
    data_empirical,
    n_xa_samples,
    simulator: callable,
    which_likelihood_calculator="gaussian",
) -> np.ndarray:
    """Picklable function at module level.

    args:
        sam_id: (int) a dummy sample index  which is required in the main `run_tmcmc` workflow

        param_vec (array-like): indeed a 1d epistemic_param_vec, the parameter vector, i.e. epistemic_domain


    note:
        Gaussian approximate likelihood function is implemented here. Other likelihood functions such as
        "pseudo" and "vae" are underway. `sam_id` is not used here, but it is required in the main `run_tmcmc` workflow
    """

    if which_likelihood_calculator == "gaussian":
        llhd_calculator = gaussian_likelihood_fun
    elif which_likelihood_calculator == "pseudo":
        llhd_calculator = "pseudo"
    elif which_likelihood_calculator == "vae":
        llhd_calculator = "vae"
    else:
        raise ValueError(f"Unknown likelihood function: {which_likelihood_calculator}")

    pseudo_log_likelihoods = llhd_calculator(
        params=param_vec,
        data=data_empirical,
        simulator=simulator,
        n_xa_samples=n_xa_samples,
    )
    return np.sum(pseudo_log_likelihoods, axis=-1)


def gaussian_likelihood_fun(
    params, data, simulator, n_xa_samples, dissimilarity_metric="wasserstein"
):
    """Computes the Gaussian log-likelihood of the data given the model and parameters.

    args:
        params: Parameters for the black-box model.

        data: Empirical data (numpy array of shape (time_steps, features, samples)).

        simulator: A function that generates model response samples given parameters.
            Here response may be time series data of shape (time_steps, features, n_xa_samples), or
            maybe some performance metrics extracted from the time series data.

        n_xa_samples: Number of samples to generate from the simulator.

    returns:
        log_likelihood: The Gaussian log-likelihood of the data given the model.
    """

    # Generate samples from the black box model using params
    mod_sam = simulator(params, n_xa_samples)

    # Compute global dissimilarity
    if dissimilarity_metric == "mse":
        # Mean Squared Error
        dissimilarity = mean_squared_error(data.flatten(), mod_sam.flatten())

        # time series case
        # data_expanded = np.expand_dims(data, axis=-1)  # Shape: (60, 6, 100, 1)
        # mod_sam_expanded = np.expand_dims(mod_sam, axis=-2)  # Shape: (60, 6, 1, 10)
        # dissimilarity = np.mean((data_expanded - mod_sam_expanded) ** 2)

    elif dissimilarity_metric == "wasserstein":
        dissimilarity = wasserstein_distance(data.flatten(), mod_sam.flatten())

    elif dissimilarity_metric == "correlation":
        corr = np.corrcoef(data.flatten(), mod_sam.flatten())[0, 1]
        dissimilarity = 1 - corr  # Correlation-based distance

    else:
        raise ValueError(f"Unknown dissimilarity metric: {dissimilarity_metric}")

    # Convert dissimilarity to likelihood
    eps_scale = 1e-3  # scaling factor for likelihood
    log_likelihood = -((dissimilarity / eps_scale) ** 2)  # np.exp(-dissimilarity)

    return log_likelihood


# * ----------------------------------- helper functions


def get_top_samples(trace, top_num=20):
    """Get the top posterior samples based on likelihood values."""
    posterior_samples = trace[-1].samples  # shape (N, 8)
    likelihoods = trace[-1].log_likelihoods  # shape (N,)

    # Select top 20% samples by likelihood
    top_indices = np.argsort(likelihoods)[-top_num:]
    top_samples = posterior_samples[top_indices]
    return top_samples


def get_posterior_samples(trace: list[Stage]) -> NDArray:
    """Get the posterior particles of the last stage from the trace.

    args:
        trace (list[Stage]): trace object.
    """
    posterior_samples = trace[-1].samples  # shape (N, 8)
    return posterior_samples


def hdi_1d(x: NDArray, cred_mass=0.95) -> tuple[float, float]:
    """Obtain the highest density interval (HDI) from particles

    args:
        x (NDArray): 1D array of samples.
        cred_mass (float): Credible mass for the HDI.

    returns:
        tuple: Lower and upper bounds of the HDI.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan)

    x = np.sort(x)
    n = x.size
    m = int(np.floor(cred_mass * n))
    if m < 1:
        return (np.nan, np.nan)

    low = x[: n - m]
    high = x[m:]
    widths = high - low
    j = np.argmin(widths)

    return (low[j], high[j])


def get_hdi_bounds(
    data: pd.DataFrame | NDArray, levels=(0.99, 0.95, 0.20, 0.10, 0.05)
) -> pd.DataFrame:
    """Compute HDI bounds for each column in `data`

    args:
        data (pd.DataFrame | NDArray): posterior samples (shape: n_samples x n_vars).
        levels (tuple): A tuple of high density intervalc levels.

    returns:
        pd.DataFrame: DataFrame with HDI bounds for each column and level.
    """
    # Normalize input
    if isinstance(data, pd.DataFrame):
        values = data.to_numpy()
        columns = data.columns
    else:
        values = np.asarray(data)
        if values.ndim != 2:
            raise ValueError("Input array must be 2D (n_samples, n_columns)")
        columns = [f"col_{i}" for i in range(values.shape[1])]

    out = {}

    for j, col in enumerate(columns):
        col_vals = values[:, j]
        for lvl in levels:
            lo, hi = hdi_1d(col_vals, cred_mass=lvl)
            p = int(round(lvl * 100))
            out.setdefault(f"hdi_{p}_low", {})[col] = lo
            out.setdefault(f"hdi_{p}_high", {})[col] = hi

    return pd.DataFrame(out).rename_axis("column").reset_index()


def transform_old_trace_to_new(trace: list[list]) -> list[Stage]:
    """Transform old trace format to new Stage class format.

    args:
        trace (list[list]): Old trace instance which is list of lists, where each inner list contains:
            [Sm, Lm, Wm_n, ESS, beta, Smcap]

    returns:
        list[Stage]: New trace instance which is list of Stage class instances.
    """
    return [Stage(*i) for i in trace]
