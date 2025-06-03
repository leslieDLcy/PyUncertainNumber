import numpy as np
from geneticalgorithm import geneticalgorithm as ga


class GA:
    """high-level interface for the genetic algorithm"""

    # TODO add a descriptor for `task`
    def __init__(self, f, task, dimension, varbound):
        """
        args:
            f (callable): function to be optimized
        """
        self.f = f
        self.task = task
        self.dimension = dimension
        self.varbound = varbound
        self.setup()

    def setup(self):
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
        """run the genetic algorithm"""
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
