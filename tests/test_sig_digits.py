# -*- coding: utf-8 -*-

import pyuncertainnumber as pun
from pyuncertainnumber.characterisation.utils import sgnumber

print("\n\nSignificant digits in pun")


def unumber(s):
    return pun.I(s).intervals


def outs(s):
    return "{:>10}\t".format(s)


s = "200.000"
print(outs(s), unumber(s), "\tshould be", 199.9995, 200.0005)
s = "200.00"
print(outs(s), unumber(s), "\tshould be", 199.995, 200.005)
s = "200.0"
print(outs(s), unumber(s), "\tshould be", 199.95, 200.05)
s = "200."
print(outs(s), unumber(s), "\tshould be", 199.5, 200.5)
s = "20.e1"
print(outs(s), unumber(s), "\tshould be", 195, 205)
s = "1.23"
print(
    outs(s), unumber(s), "\tshould be", 1.225, 1.235
)  # extra 9s are a feature of Python
s = "1.2300"
print(outs(s), unumber(s), "\tshould be", 1.22995, 1.23005)
s = "12.3"
print(outs(s), unumber(s), "\tshould be", 12.25, 12.35)
s = "1.23e8"
print(outs(s), unumber(s), "\tshould be", 122500000.0, 123500000.0)
s = "12.3e2"
print(outs(s), unumber(s), "\tshould be", 1225.0, 1235.0)
s = "12.3e-4"
print(outs(s), unumber(s), "\tshould be", 0.001225, 0.001235)
s = "12300"
print(outs(s), unumber(s), "\tshould be", 12250.0, 12350.0)
s = "12300e4"
print(outs(s), unumber(s), "\tshould be", 122500000.0, 123500000.0)
s = "9"
print(outs(s), unumber(s), "\tshould be", 8.5, 9.5)
s = "2e2"
print(outs(s), unumber(s), "\tshould be", 150, 250)
s = "200"
print(outs(s), unumber(s), "\tshould be", 150, 250)
s = "10"
print(outs(s), unumber(s), "\tshould be", 5, 15)
s = "1000"
print(outs(s), unumber(s), "\tshould be", 500, 1500)
s = "1000000"
print(outs(s), unumber(s), "\tshould be", 5e5, 15e5)
s = "123400"
print(outs(s), unumber(s), "\tshould be", 123350, 123450)
s = "12000"
print(outs(s), unumber(s), "\tshould be", 11500, 12500)

print("\n\nInterpreting significant digits")


def outs(s):
    return "{:>10}\t".format(s)


s = "200.000"
print(outs(s), sgnumber(s))  # [ 199.9995, 200.0005]
s = "200.00"
print(outs(s), sgnumber(s))  # [ 199.995, 200.005]
s = "200.0"
print(outs(s), sgnumber(s))  # [ 199.95, 200.05]
s = "200."
print(outs(s), sgnumber(s))  # [ 199.5, 200.5]
s = "20.e1"
print(outs(s), sgnumber(s))  # [ 195, 205]
s = "1.23"
print(outs(s), sgnumber(s))  # 1.225, 1.235  # the extra nines are a 'feature' of Python
s = "1.2300"
print(outs(s), sgnumber(s))  # 1.22995, 1.23005
s = "12.3"
print(outs(s), sgnumber(s))  # 12.25, 12.35
s = "1.23e8"
print(outs(s), sgnumber(s))  # 122500000.0, 123500000.0
s = "12.3e2"
print(outs(s), sgnumber(s))  # 1225.0, 1235.0
s = "12.3e-4"
print(outs(s), sgnumber(s))  # 0.001225, 0.001235
s = "12300"
print(outs(s), sgnumber(s))  # 12250.0, 12350.0
s = "12300e4"
print(outs(s), sgnumber(s))  # 122500000.0, 123500000.0
s = "9"
print(outs(s), sgnumber(s))  # 8.5,9.5
s = "2e2"
print(outs(s), sgnumber(s))  # 150, 250
s = "200"
print(outs(s), sgnumber(s))  # 150,250
s = "10"
print(outs(s), sgnumber(s))  # 5,15
s = "1000"
print(outs(s), sgnumber(s))  # 500,1500
s = "1000000"
print(outs(s), sgnumber(s))  # 5e5,15e5
s = "123400"
print(outs(s), sgnumber(s))  # 123350,123450
s = "12000"
print(outs(s), sgnumber(s))  # 11500,12500
