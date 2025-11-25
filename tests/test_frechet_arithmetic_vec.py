"""to verify the arithmetic operations on Frechet ops on a new vectorised implementation

hint: this verifies that new vectorised operations match the old reference implementation
"""

from pyuncertainnumber import pba
from pyuncertainnumber.pba.pbox_abc import inspect_pbox

a = pba.normal([4, 5], 1)  # positive
# b = pba.uniform([3, 4], [7, 8])  # positive

# a = pba.normal([1, 2], 1)  # straddles zero
b = pba.uniform([-3, -2], [3, 4])  # straddles 0


### + ###
c_ref_add = a + b
c_new_add = a.vec_add(b, dependency="f")
assert c_ref_add == c_new_add, "Addition operation not matching"

### - ###
c_ref_sub = a - b
c_new_sub = a.vec_sub(b, dependency="f")
assert c_ref_sub == c_new_sub, "Subtraction operation not matching"

### * ###
c_ref_mul = a * b
c_new_mul = a.vec_mul(b, dependency="f")
assert c_ref_mul == c_new_mul, "Multiplication operation not matching"

### / ###
# c_ref_div = a / b
# c_new_div = a.vec_div(b, dependency="f")
# assert c_ref_div == c_new_div, "Division operation not matching"

print("-------test over-------")
