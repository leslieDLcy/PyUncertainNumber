""" Hyperparameters for the pba """

from dataclasses import dataclass

''' notes: '''

@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:

    steps = 200

    # @property
    # # template for property
    # def sth(self):
    #     """ template for property"""
    #     return int(round(self.patch_window_seconds / self.stft_hop_seconds))
    




