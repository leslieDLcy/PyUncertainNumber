""" Constructors for Dempester-Shafer structures. """
import numpy as np
from .intervalOperators import make_vec_interval
from collections import namedtuple
from .utils import reweighting, stacking, plot_DS_structure

dempstershafer_element = namedtuple(
    'dempstershafer_element', ['interval', 'weight'])


class DempsterShafer:
    """ Class for Dempester-Shafer structures. """

    def __init__(self, intervals, masses: list[float]):
        self._intrep = np.array(intervals)
        self._intervals = make_vec_interval(intervals)
        self._masses = np.array(masses)

    def _create_DSstructure(self):
        return [dempstershafer_element(i, m) for i, m in zip(self._intervals, self._masses)]

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

    def display(self, style='box'):
        intervals, masses = self.disassemble()
        match style:
            case 'box':

                stacking(intervals, masses, display=True)
            case 'interval':
                plot_DS_structure(intervals, masses)


def mixture_ds(l_ds, display=False):
    """ mixture operation for DS structure """

    intervals = np.concatenate([ds.disassemble()[0] for ds in l_ds], axis=0)
    # TODO check the duplicate intervals
    # assert sorted(intervals) == np.unique(intervals), "intervals replicate"
    masses = reweighting([ds.disassemble()[1] for ds in l_ds])
    return DempsterShafer(intervals, masses)
    # below is to return the mixture as in a pbox
    # return stacking(intervals, masses, display=display)
