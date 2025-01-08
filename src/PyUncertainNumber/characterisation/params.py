""" Hyperparameters for the wind turbine loading case """

from dataclasses import dataclass

""" notes: """


@dataclass(frozen=True)  # Instances of this class are immutable.
class Params:

    result_path = "./results/"
    steps = 500
    hw = 0.5  # default half-width during an interval instantiation via PM method

    @property
    # template for property
    def sth(self):
        """template for property"""
        return int(round(self.patch_window_seconds / self.stft_hop_seconds))
