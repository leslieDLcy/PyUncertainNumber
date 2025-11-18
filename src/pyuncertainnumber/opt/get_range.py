from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyuncertainnumber import Interval

from .bo import BayesOpt
from .ga import GA
import numpy as np


def get_range_BO(
    f: callable,
    design_bounds: list | np.ndarray,
    acquisition_function: str = "UCB",
    verbose: bool = False,
    **kwargs,
) -> tuple[Interval, dict]:
    """Compute the range of a black-box function using BayesOpt with vectorised or iterable function signature

    args:
        f (callable): function as the objective to be optimized, with vectorised or iterable function signature

        design_bounds (list | np.ndarray): nested list or 2D array, containing the lower and upper bounds for each dimension; e.g. [[0, 1], [0, 1]] for 2D input space.

        acquisition_function (str or callable, optional): the acquisition function to be used, e.g. 'UCB', 'EI', 'PI'. If None, defaults to 'UCB'.

        verbose (Boolean): if True, prints the optimization progress

        **kwargs: additional keyword arguments for the BayesOpt class. For example, one can pass 'num_explorations', 'num_iterations', etc.

    tip:
        for a less verbose output, use (convergence_curve=False, progress_bar=False)

    return:
        Tuple[Interval, dict]: A tuple containing:
            - response_itvl: The interval of the minimum and maximum from the optimization of the black-box function.
            - opt_hint: A dictionary with optimal input points for the minimum and maximum values.

    """
    from ..pba.intervals.number import Interval

    if not verbose:  # quiet mode
        V = 0
    else:
        V = 1

    min_task = BayesOpt(
        f=f,
        task="minimisation",
        design_bounds=design_bounds,
        acquisition_function=acquisition_function,
        **kwargs,
    )
    min_task.run(verbose=V)
    min_target = min_task.optimal_target

    max_task = BayesOpt(
        f=f,
        task="maximisation",
        design_bounds=design_bounds,
        acquisition_function=acquisition_function,
        **kwargs,
    )

    max_task.run(verbose=V)
    max_target = max_task.optimal_target

    # return 1: the interval of min and max
    response_itvl = Interval(min_target, max_target)

    # return 2: the mapping associated with the optimisation
    opt_hint = {
        "min": min_task.optimal,
        "max": max_task.optimal,
    }
    return response_itvl, opt_hint


def get_range_BO_raw(
    f: callable,
    design_bounds: list,
    acquisition_function="UCB",
    verbose=False,
    **kwargs,
):
    """Compute the range of a black-box function using BayesOpt with arguments-signature function;

    args:
        f: callable function to be optimized

        dimension (int): the number of dimensions of the input space

        design_bounds (list): each tuple contains the lower and upper bounds for each dimension

        acquisition_function (str or callable, optional): the acquisition function to be used, e.g. 'UCB', 'EI', 'PI'. If None, defaults to 'UCB'.

        verbose (Boolean): if True, prints the optimization progress

        **kwargs: additional keyword arguments for the BayesOpt class

    tip:
        for a less verbose output, use (convergence_curve=False, progress_bar=False)

    return:
        Tuple[Interval, dict]: A tuple containing:
            - response_itvl: The interval of the minimum and maximum from the optimization of the black-box function.
            - opt_hint: A dictionary with optimal input points for the minimum and maximum values.

    """
    from ..pba.intervals.number import Interval

    if not verbose:  # quiet mode
        V = 0
    else:
        V = 1

    min_task = BayesOpt(
        f=f,
        task="minimisation",
        design_bounds=design_bounds,
        acquisition_function=acquisition_function,
        **kwargs,
    )
    min_task.run(verbose=V)
    min_target = min_task.optimal_target

    max_task = BayesOpt(
        f=f,
        task="maximisation",
        design_bounds=design_bounds,
        acquisition_function=acquisition_function,
        **kwargs,
    )

    max_task.run(verbose=V)
    max_target = max_task.optimal_target

    # return 1: the interval of min and max
    response_itvl = Interval(min_target, max_target)

    # return 2: the mapping associated with the optimisation
    opt_hint = {
        "min": min_task.optimal,
        "max": max_task.optimal,
    }
    return response_itvl, opt_hint


def get_range_GA(
    f: callable, dimension: int, varbound, algorithm_param=None, verbose=False, **kwargs
):
    """compute the range of the black-box function using GA

    args:
        varbound (np.ndarray): The variable bounds for the optimization.

    return:
        Tuple[Interval, dict]: A tuple containing:
            - response_itvl: The interval of the minimum and maximum from the optimization of the black-box function.
            - opt_hint: A dictionary with optimal input points for the minimum and maximum values.

    note:
        It's suggested to use `EpistemicDomain` which facilitates the specification of varbound.
    """

    from ..pba.intervals.number import Interval

    if not verbose:
        kwargs["convergence_curve"] = False
        kwargs["progress_bar"] = False

    min_task = GA(f, task="minimisation", dimension=dimension, varbound=varbound)
    min_task.run(algorithm_param=algorithm_param, **kwargs)
    min_target = min_task.optimal_target

    max_task = GA(f, task="maximisation", dimension=dimension, varbound=varbound)
    max_task.run(algorithm_param=algorithm_param, **kwargs)
    max_target = max_task.optimal_target

    # return 1: the interval of min and max
    response_itvl = Interval(min_target, max_target)

    # return 2: the mapping associated with the optimisation
    opt_hint = {
        "min": min_task.optimal,
        "max": max_task.optimal,
    }

    return response_itvl, opt_hint
