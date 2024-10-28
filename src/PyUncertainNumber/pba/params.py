""" Hyperparameters for the pba """

from dataclasses import dataclass
import numpy as np

''' notes: '''

@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:

    steps = 200
    
    # the percentiles
    p_values = np.linspace(0.0001, 0.9999, steps)

    # @property
    # # template for property
    # def sth(self):
    #     """ template for property"""
    #     return int(round(self.patch_window_seconds / self.stft_hop_seconds))
    




