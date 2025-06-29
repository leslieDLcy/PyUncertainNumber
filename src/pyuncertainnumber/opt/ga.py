import numpy as np
from geneticalgorithm import geneticalgorithm as ga


class GA:
    """Genetic Algorithm class

    args:
        f (callable): the target function to be optimised, should have a single argument

        task (str): either 'minimisation' or 'maximisation'

        dimension (int): the dimension of the design space, i.e. the number of parameters

        varbound (np.ndarry): the bounds for the design space, e.g. 'np.array([[-2, 10]])'

    example:
        >>> import numpy as np
        >>> from pyuncertainnumber.opt.ga import GA
        >>> def black_box_function(x):
        ...     return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2 / 10) + 1 / (x**2 + 1)
        >>> ga = GA(f=black_box_function, task='maximisation', dimension=1, varbound=np.array([[-2, 10]]))
        >>> ga.run()  # the progress bar will be shown as side effect
        >>> print(ga.optimal)  # get the optimal parameters and target value

    .. admonition:: Implementation

        The range of the design space is defined by `varbound`, which is a 2D numpy array with shape (n, 2), where n is the number of parameters.
        This is a different signature compared to the Bayesian Optimisation class, which uses a dictionary for bounds.
        For consistency, it is recommended to use the class `EpistemicDomain.to_varbound()` to automatically take care of the format of the bounds.

        example:
            >>> from pyuncertainnumber import pba, EpistemicDomain
            >>> ed = EpistemicDomain(pba.I(-5, 5), pba.I(-5, 5))
            >>> ga = GA(
            ...     f=foo,
            ...     task='maximisation',
            ...     dimension=2,
            ...     varbound=ed.to_GA_varBounds()  # the trick
            ... )

    .. seealso::

        :class:`pyuncertainnumber.propagation.epistemic_uncertainty.helper.EpistemicDomain`: the utility tool for setting up the epistemic domain.
    """

    # TODO add a descriptor for `task`
    def __init__(self, f: callable, task: str, dimension: int, varbound):

        self.f = f
        self.task = task
        self.dimension = dimension
        self.varbound = varbound
        self.setup()

    def setup(self):
        """Objective direction setup"""

        if self.task == "maximisation":
            self.flip_f = lambda *args, **kwargs: -self.f(*args, **kwargs)
            self._f = self.flip_f
        elif self.task == "minimisation":
            self._f = self.f
        else:
            raise ValueError("task should be either 'maximisation' or 'minimisation'")

    def get_results(self):
        """display the results of the optimization"""
        # direct result from GA
        self.output = self.model.output_dict.copy()

        if self.task == "maximisation":
            self.output["function"] = np.absolute(self.output["function"])
        elif self.task == "minimisation":
            pass

        self._optimal_dict = {}
        self._optimal_dict["xc"] = self.output["variable"]
        self._optimal_dict["target"] = self.output["function"]
        self._all_results = self.output
        # hint: self.model.output_dict["variable"], self.model.output_dict["function"]

    def run(self, algorithm_param=None, **kwargs):
        """run the genetic algorithm

        args:
            algorithm_param (dict): the parameters for the genetic algorithm; if None, the default parameters will be used.
            convergence_curve (Boolean): whether to return the convergence curve, default is True
            progress_bar (Boolean): whether to show the progress bar, default is True
        """
        if algorithm_param is not None:
            self.model = ga(
                function=self._f,
                dimension=self.dimension,
                variable_type="real",
                variable_boundaries=self.varbound,
                algorithm_parameters=self.algorithm_param,
                function_timeout=int(1e6),
                **kwargs,
            )
        else:
            self.model = ga(
                function=self._f,
                dimension=self.dimension,
                variable_type="real",
                variable_boundaries=self.varbound,
                function_timeout=int(1e6),
                **kwargs,
            )
        self.model.run()
        self.get_results()

    @property
    def optimal(self):
        return self._optimal_dict

    @property
    def optimal_xc(self):
        """return the design variable that gives the optimal function value"""
        try:
            return self.model.output_dict["variable"].item()
        except AttributeError:
            raise AttributeError("You need to run the model first.")

    @property
    def optimal_target(self):
        """return the optimal target value"""
        return self._optimal_dict["target"]
