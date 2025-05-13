from .bo import BayesOpt
from .ga import GA


def get_range_BO(f, xc_bounds, **kwargs):
    """get the range of the black-box function using BayesOpt

    note:
        for a less verbose output, use (convergence_curve=False, progress_bar=False)
    """
    min_task = BayesOpt(f=f, task="minimisation", xc_bounds=xc_bounds, **kwargs)
    min_task.run()
    min = min_task.optimal_target

    max_task = BayesOpt(f=f, task="maximisation", xc_bounds=xc_bounds, **kwargs)
    max_task.run()
    min = max_task.optimal_target
    return min, max


def get_range_GA(f, dimension, varbound, algorithm_param=None, **kwargs):
    """get the range of the black-box function using GA"""

    min_task = GA(f, task="minimisation", dimension=dimension, varbound=varbound)
    min_task.run(algorithm_param=algorithm_param, **kwargs)
    min = min_task.optimal_target

    max_task = GA(f, task="maximisation", dimension=dimension, varbound=varbound)
    max_task.run(algorithm_param=algorithm_param, **kwargs)
    max = max_task.optimal_target

    return min, max
