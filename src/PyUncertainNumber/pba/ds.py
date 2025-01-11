""" Constructors for Dempester-Shafer structures. """
import numpy as np
from .intervalOperators import make_vec_interval
from collections import namedtuple
from .utils import reweighting, stacking, plot_DS_structure
from .constructors import pbox_fromDiscreteF

dempstershafer_element = namedtuple(
    'dempstershafer_element', ['interval', 'mass'])


class DempsterShafer:
    # TODO add a new constructor see C report P. 76
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

    def disassemble(self,):
        return self._intrep, self._masses

    def display(self, style='box'):
        # TODO cannot take kwargs (such as title='') yet. to be fixed
        # TODO slightly different call signature compared to pbox & Interval
        intervals, masses = self.disassemble()
        match style:
            case 'box':
                stacking(intervals, masses, display=True)
            case 'interval':
                plot_DS_structure(intervals, masses)

    def to_pbox(self):
        intervals, masses = self.disassemble()
        return stacking(intervals, masses, return_pbox=True)


def mixture_ds(l_ds, display=False):
    """ mixture operation for DS structure """

    intervals = np.concatenate([ds.disassemble()[0] for ds in l_ds], axis=0)
    # TODO check the duplicate intervals
    # assert sorted(intervals) == np.unique(intervals), "intervals replicate"
    masses = reweighting([ds.disassemble()[1] for ds in l_ds])
    return DempsterShafer(intervals, masses)
    # below is to return the mixture as in a pbox
    # return stacking(intervals, masses, display=display)
