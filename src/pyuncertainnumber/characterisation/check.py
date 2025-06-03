from __future__ import annotations
from typing import TYPE_CHECKING
from .utils import initial_list_checking
from ..pba.intervals import Interval

""" This module is for checking the logic for the instantiation and propagation of the Uncertain Number object"""

""" the specification detail we need for instantiating the UN distribution or pbox object, see below 
fullname will be `distribution_specification`
"""

# Dist_spec = namedtuple("Dist_spec", ["x", "y"])

# ['uniform', [(0,1),(1,2)]]


class DistributionSpecification:
    """an attempt to double check the user specification for a pbox or dist

    note:
        - canonical form: ['gaussian', ([0,1], [1,2])]
    # TODO: unfinished logic"""

    def __init__(self, dist_family: str, dist_params: tuple):
        self.dist_family = dist_family
        self.dist_params = dist_params
        self.tell_i_flag()

    def tell_i_flag(self):
        """boolean about if imprecise specification"""

        if isinstance(self.dist_params[0], float | int):
            self._i_flag = False
        elif isinstance(self.dist_params[0], Interval | tuple | list):
            self._i_flag = True
        else:
            raise ValueError("The disribution parameters are not clear")

    @property
    def i_flag(self):
        return self._i_flag

    def get_specification(self):
        if isinstance(self.dist_params, str):  # string parsing
            return [self.dist_family, initial_list_checking(self.dist_params)]
        elif isinstance(self.dist_params, list):
            return [self.dist_family, self.dist_params]
        else:
            return [self.dist_family, self.dist_params]
