""" Constructors for Dempester-Shafer structures. """
import numpy as np
from .intervalOperators import make_vec_interval


class DempsterShafer:
    """ Class for Dempester-Shafer structures. """

    def __init__(self, intervals, masses: list[float]):
        self._intrep = np.array(intervals)
        self._intervals = make_vec_interval(intervals)
        self._masses = np.array(masses)

    def _create_DSstructure(self):
        return [(i, m) for i, m in zip(self._intervals, self._masses)]

    @property
    def structure(self):
        return self._create_DSstructure()

    @property
    def intervals(self):
        return self._intervals

    @property
    def masses(self):
        return self._masses

    # def __add__(self, other):
    #     if isinstance(other, DempsterShafer):
    #         lo = np.concatenate((self.intervals.lo, other.intervals.lo))
    #         hi = np.concatenate((self.intervals.hi, other.intervals.hi))
    #         masses = np.concatenate((self.masses, other.masses))

    #     else:
    #         raise ValueError("Can only add DempsterShafer objects together")

    def disassemble(self,):
        return self._intrep, self._masses
