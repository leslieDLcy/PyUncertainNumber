""" Hyperparameters for the pba """

from dataclasses import dataclass
import numpy as np

''' notes: '''

@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:

    steps = 200
    many = 2000
    # the percentiles
    p_values = np.linspace(0.0001, 0.9999, steps)

    p_lboundary = 0.0001
    p_hboundary = 0.9999
    

    # sample data for a go-to example
    s = [4.02, 4.07, 4.25, 4.32, 4.36, 4.45, 4.47, 
         4.57, 4.58, 4.62, 4.68, 4.71, 4.72, 4.79, 
         4.85, 4.86, 4.88, 4.90, 5.08, 5.09, 5.29, 
         5.30, 5.40, 5.44, 5.59, 5.59, 5.70, 5.89, 
         5.89, 6.01]



     # by default
    scott_hedged_interpretation = {}
    

    # user-defined 
    user_hedged_interpretation = {}




    # @property
    # # template for property
    # def sth(self):
    #     """ template for property"""
    #     return int(round(self.patch_window_seconds / self.stft_hop_seconds))
    




