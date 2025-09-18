# read version from installed package
from importlib.metadata import version

__version__ = version(__name__)


from pyuncertainnumber.characterisation.uncertainNumber import UncertainNumber as UN
from pyuncertainnumber.characterisation.uncertainNumber import *

# * --------------------- pba ---------------------*#
import pyuncertainnumber.pba as pba
from pyuncertainnumber.pba.pbox_free import *
from pyuncertainnumber.characterisation.stats import fit


# * --------------------- Pbox ---------------------*#
from pyuncertainnumber.pba.pbox_abc import Pbox

# * --------------------- Interval ---------------------*#
from pyuncertainnumber.pba.intervals.number import Interval
from pyuncertainnumber.pba.intervals.intervalOperators import make_vec_interval
from pyuncertainnumber.pba.intervals import intervalise
from pyuncertainnumber.propagation.epistemic_uncertainty.helper import EpistemicDomain


# * --------------------- hedge---------------------*#
from pyuncertainnumber.nlp.language_parsing import hedge_interpret


# * --------------------- cbox ---------------------*#
from pyuncertainnumber.pba.cbox import infer_cbox, infer_predictive_distribution


# * --------------------- DempsterShafer ---------------------*#
from pyuncertainnumber.pba.dss import dempstershafer_element, DempsterShafer


# * --------------------- Dependency ---------------------*#
from pyuncertainnumber.pba.dependency import Dependency

# * ---------------------  aggregation ---------------------*#
from pyuncertainnumber.pba.aggregation import *

# * ---------------------  propagation ---------------------*#
from pyuncertainnumber.propagation.epistemic_uncertainty.b2b import b2b
from pyuncertainnumber.propagation.p import Propagation
from pyuncertainnumber.propagation.taylor_expansion import taylor_expansion_method

# * ---------------------  utils ---------------------*#
from pyuncertainnumber.pba import inspect_un
