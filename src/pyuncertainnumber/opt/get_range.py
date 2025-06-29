from .bo import BayesOpt
from .ga import GA


def get_range_BO(
    f, dimension, xc_bounds, acquisition_function, verbose=False, **kwargs
):
    """compute the range of a black-box function using BayesOpt

    implementation:
        for a less verbose output, use (convergence_curve=False, progress_bar=False)

    return:
        - Interval: ``response_itvl``, the interval of the minimum and maximum out of the optimisation given black-box function
        - dict: ``opt_hint``, the mapping associated with the optimisation, containing the optimal points for min and max
    """
    from ..pba.intervals.number import Interval

    if not verbose:  # quiet mode
        V = 0

    min_task = BayesOpt(
        f=f,
        dimension=dimension,
        task="minimisation",
        xc_bounds=xc_bounds,
        acquisition_function=acquisition_function,
        **kwargs,
    )
    min_task.run(verbose=V)
    min_target = min_task.optimal_target

    max_task = BayesOpt(
        f=f,
        dimension=dimension,  # type: ignore
        task="maximisation",
        xc_bounds=xc_bounds,
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


def get_range_GA(f, dimension, varbound, algorithm_param=None, verbose=False, **kwargs):
    """compute the range of the black-box function using GA"""

    if not verbose:
        kwargs["convergence_curve"] = False
        kwargs["progress_bar"] = False

    min_task = GA(f, task="minimisation", dimension=dimension, varbound=varbound)
    min_task.run(algorithm_param=algorithm_param, **kwargs)
    min = min_task.optimal_target

    max_task = GA(f, task="maximisation", dimension=dimension, varbound=varbound)
    max_task.run(algorithm_param=algorithm_param, **kwargs)
    max = max_task.optimal_target

    return min, max
