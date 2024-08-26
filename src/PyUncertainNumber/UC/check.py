from dataclasses import dataclass, field
from typing import Tuple, List
from PyUncertainNumber.UC.utils import initial_list_checking

""" This module is for checking the logic for the instantiation and propagation of the Uncertain Number object"""

""" the specification detail we need for instantiating the UN distribution or pbox object, see below 
fullname will be `distribution_specification`
"""

# Dist_spec = namedtuple("Dist_spec", ["x", "y"])

# ['uniform', [(0,1),(1,2)]]


class DistributionSpecification:
    """an attempt to double check the user specification for a pbox or dist
    # TODO: unfinished logic"""

    def __init__(self, dist_family, dist_params):
        self.dist_family = dist_family
        self.dist_params = dist_params

    def get_specification(self):
        if isinstance(self.dist_params, str):
            return [self.dist_family, initial_list_checking(self.dist_params)]
        elif isinstance(self.dist_params, list):
            return [self.dist_family, self.dist_params]
