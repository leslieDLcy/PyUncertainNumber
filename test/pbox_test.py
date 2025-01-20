import pyuncertainnumber as pun
import numpy as np
from pyuncertainnumber import pba

# *  ---------------------construction---------------------*#
print(pun.norm([2, 3], [0.1]))
print(pun.I([2, 3]))
# *  ---------------------aggregation---------------------*#

from pyuncertainnumber import stochastic_mixture

lower_endpoints = np.random.uniform(-0.5, 0.5, 7)
upper_endpoints = np.random.uniform(0.5, 1.5, 7)
m_weights = [0.1, 0.1, 0.25, 0.15, 0.1, 0.1, 0.2]
# a list of nInterval objects
nI = [pba.I(couple) for couple in zip(lower_endpoints, upper_endpoints)]
pbox_mix = stochastic_mixture(nI, weights=m_weights, display=True, return_type="pbox")
print("the result of the mixture operation")
print(pbox_mix)

# *  ---------------------arithmetic---------------------*#

### pba level ###
a = pba.I([2, 3])  # an interval
# _ = a.display(style="band", title="Interval [2,3]")
b = pba.norm(0, 1)  # a precise distribution
# _ = b.display(title="$N(0, 1)$")
t = a + b
print(t)


### UN level ###
a = pun.norm(0, 1)
b = pun.norm(2, 3)
t = a + b
print(t)
