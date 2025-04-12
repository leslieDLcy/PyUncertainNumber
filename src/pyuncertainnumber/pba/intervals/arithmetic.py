"""
--------------------------
Created Feb 2022
Marco De Angelis
github.com/marcodeangelis
MIT License
--------------------------
"""

import numpy


def multiply(s, o):
    s_lo, s_hi, o_lo, o_hi = s.lo, s.hi, o.lo, o.hi
    if s.scalar & o.scalar:
        if (s_lo >= 0) & (o_lo >= 0):  # A+ B+
            l, h = s_lo * o_lo, s_hi * o_hi
        if (s_lo >= 0) & ((o_lo < 0) & (o_hi > 0)):  # A+ B0
            l, h = s_hi * o_lo, s_hi * o_hi
        if (s_lo >= 0) & (o_hi <= 0):  # A+ B-
            l, h = s_hi * o_lo, s_lo * o_hi
        if ((s_lo < 0) & (s_hi > 0)) & (o_lo >= 0):  # A0 B+
            l, h = s_lo * o_hi, s_hi * o_hi
        if ((s_lo < 0) & (s_hi > 0)) & ((o_lo < 0) & (o_hi > 0)):  # A0 B0
            l = numpy.min((s_lo * o_hi, s_hi * o_lo, s_lo * o_lo, s_hi * o_hi), axis=0)
            h = numpy.max((s_lo * o_lo, s_hi * o_hi, s_lo * o_hi, s_hi * o_lo), axis=0)
        if ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0):  # A0 B-
            l, h = s_hi * o_lo, s_lo * o_lo
        if (s_hi <= 0) & (o_lo >= 0):  # A- B+
            l, h = s_lo * o_hi, s_hi * o_lo
        if (s_hi <= 0) & ((o_lo < 0) & (o_hi > 0)):  # A- B0
            l, h = s_lo * o_hi, s_lo * o_lo
        if (s_hi <= 0) & (o_hi <= 0):  # A- B-
            l, h = s_hi * o_hi, s_lo * o_lo
    elif s_lo.shape == o_lo.shape:
        l, h = numpy.empty(s_lo.shape), numpy.empty(s_lo.shape)
        pp = (s_lo >= 0) & (o_lo >= 0)  # A+ B+
        l[pp] = s_lo[pp] * o_lo[pp]
        h[pp] = s_hi[pp] * o_hi[pp]
        pz = (s_lo >= 0) & ((o_lo < 0) & (o_hi > 0))  # A+ B0
        l[pz] = s_hi[pz] * o_lo[pz]
        h[pz] = s_hi[pz] * o_hi[pz]
        pn = (s_lo >= 0) & (o_hi <= 0)  # A+ B-
        l[pn] = s_hi[pn] * o_lo[pn]
        h[pn] = s_lo[pn] * o_hi[pn]
        zp = ((s_lo < 0) & (s_hi > 0)) & (o_lo >= 0)  # A0 B+
        l[zp] = s_lo[zp] * o_hi[zp]
        h[zp] = s_hi[zp] * o_hi[zp]
        zz = ((s_lo < 0) & (s_hi > 0)) & ((o_lo < 0) & (o_hi > 0))  # A0 B0
        l[zz] = numpy.min(
            (
                s_lo[zz] * o_hi[zz],
                s_hi[zz] * o_lo[zz],
                s_lo[zz] * o_lo[zz],
                s_hi[zz] * o_hi[zz],
            ),
            axis=0,
        )
        h[zz] = numpy.max(
            (
                s_lo[zz] * o_lo[zz],
                s_hi[zz] * o_hi[zz],
                s_lo[zz] * o_hi[zz],
                s_hi[zz] * o_lo[zz],
            ),
            axis=0,
        )
        zn = ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0)  # A0 B-
        l[zn] = s_hi[zn] * o_lo[zn]
        h[zn] = s_lo[zn] * o_lo[zn]
        np = (s_hi <= 0) & (o_lo >= 0)  # A- B+
        l[np] = s_lo[np] * o_hi[np]
        h[np] = s_hi[np] * o_lo[np]
        nz = (s_hi <= 0) & ((o_lo < 0) & (o_hi > 0))  # A- B0
        l[nz] = s_lo[nz] * o_hi[nz]
        h[nz] = s_lo[nz] * o_lo[nz]
        nn = (s_hi <= 0) & (o_hi <= 0)  # A- B-
        l[nn] = s_hi[nn] * o_hi[nn]
        h[nn] = s_lo[nn] * o_lo[nn]
    elif s.scalar:
        l, h = numpy.empty(o_lo.shape), numpy.empty(o_lo.shape)
        pp = (s_lo >= 0) & (o_lo >= 0)  # A+ B+
        l[pp] = s_lo * o_lo[pp]
        h[pp] = s_hi * o_hi[pp]
        pz = (s_lo >= 0) & ((o_lo < 0) & (o_hi > 0))  # A+ B0
        l[pz] = s_hi * o_lo[pz]
        h[pz] = s_hi * o_hi[pz]
        pn = (s_lo >= 0) & (o_hi <= 0)  # A+ B-
        l[pn] = s_hi * o_lo[pn]
        h[pn] = s_lo * o_hi[pn]
        zp = ((s_lo < 0) & (s_hi > 0)) & (o_lo >= 0)  # A0 B+
        l[zp] = s_lo * o_hi[zp]
        h[zp] = s_hi * o_hi[zp]
        zz = ((s_lo < 0) & (s_hi > 0)) & ((o_lo < 0) & (o_hi > 0))  # A0 B0
        l[zz] = numpy.min(
            (s_lo * o_hi[zz], s_hi * o_lo[zz], s_lo * o_lo[zz], s_hi * o_hi[zz]), axis=0
        )
        h[zz] = numpy.max(
            (s_lo * o_lo[zz], s_hi * o_hi[zz], s_lo * o_hi[zz], s_hi * o_lo[zz]), axis=0
        )
        zn = ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0)  # A0 B-
        l[zn] = s_hi * o_lo[zn]
        h[zn] = s_lo * o_lo[zn]
        np = (s_hi <= 0) & (o_lo >= 0)  # A- B+
        l[np] = s_lo * o_hi[np]
        h[np] = s_hi * o_lo[np]
        nz = (s_hi <= 0) & ((o_lo < 0) & (o_hi > 0))  # A- B0
        l[nz] = s_lo * o_hi[nz]
        h[nz] = s_lo * o_lo[nz]
        nn = (s_hi <= 0) & (o_hi <= 0)  # A- B-
        l[nn] = s_hi * o_hi[nn]
        h[nn] = s_lo * o_lo[nn]
    elif o.scalar:
        l, h = numpy.empty(s_lo.shape), numpy.empty(s_lo.shape)
        pp = (s_lo >= 0) & (o_lo >= 0)  # A+ B+
        l[pp] = s_lo[pp] * o_lo
        h[pp] = s_hi[pp] * o_hi
        pz = (s_lo >= 0) & ((o_lo < 0) & (o_hi > 0))  # A+ B0
        l[pz] = s_hi[pz] * o_lo
        h[pz] = s_hi[pz] * o_hi
        pn = (s_lo >= 0) & (o_hi <= 0)  # A+ B-
        l[pn] = s_hi[pn] * o_lo
        h[pn] = s_lo[pn] * o_hi
        zp = ((s_lo < 0) & (s_hi > 0)) & (o_lo >= 0)  # A0 B+
        l[zp] = s_lo[zp] * o_hi
        h[zp] = s_hi[zp] * o_hi
        zz = ((s_lo < 0) & (s_hi > 0)) & ((o_lo < 0) & (o_hi > 0))  # A0 B0
        l[zz] = numpy.min(
            (s_lo[zz] * o_hi, s_hi[zz] * o_lo, s_lo[zz] * o_lo, s_hi[zz] * o_hi), axis=0
        )
        h[zz] = numpy.max(
            (s_lo[zz] * o_lo, s_hi[zz] * o_hi, s_lo[zz] * o_hi, s_hi[zz] * o_lo), axis=0
        )
        zn = ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0)  # A0 B-
        l[zn] = s_hi[zn] * o_lo
        h[zn] = s_lo[zn] * o_lo
        np = (s_hi <= 0) & (o_lo >= 0)  # A- B+
        l[np] = s_lo[np] * o_hi
        h[np] = s_hi[np] * o_lo
        nz = (s_hi <= 0) & ((o_lo < 0) & (o_hi > 0))  # A- B0
        l[nz] = s_lo[nz] * o_hi
        h[nz] = s_lo[nz] * o_lo
        nn = (s_hi <= 0) & (o_hi <= 0)  # A- B-
        l[nn] = s_hi[nn] * o_hi
        h[nn] = s_lo[nn] * o_lo
    return l, h


def divide(s, o):
    s_lo, s_hi, o_lo, o_hi = s.lo, s.hi, o.lo, o.hi
    other_straddle_zero = numpy.any((o_lo.flatten() <= 0) & (o_hi.flatten() >= 0))
    if other_straddle_zero:
        raise ZeroDivisionError
    if s.scalar & o.scalar:
        if (s_lo >= 0) & (o_lo > 0):  # A+ B+
            l, h = s_lo / o_hi, s_hi / o_lo
        if ((s_lo < 0) & (s_hi > 0)) & (o_lo > 0):  # A0 B+
            l, h = s_lo / o_lo, s_hi / o_lo
        if (s_hi <= 0) & (o_lo >= 0):  # A- B+
            l, h = s_lo / o_lo, s_hi / o_hi
        if (s_lo >= 0) & (o_hi <= 0):  # A+ B-
            l, h = s_hi / o_hi, s_lo / o_lo
        if ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0):  # A0 B-
            l, h = s_hi / o_hi, s_lo / o_hi
        if (s_hi <= 0) & (o_hi <= 0):  # A- B-
            l, h = s_hi / o_lo, s_lo / o_hi
    elif s_lo.shape == o_lo.shape:
        l, h = numpy.empty(s_lo.shape), numpy.empty(s_lo.shape)
        pp = (s_lo >= 0) & (o_lo > 0)  # A+ B+
        l[pp] = s_lo[pp] / o_hi[pp]
        h[pp] = s_hi[pp] / o_lo[pp]
        zp = ((s_lo < 0) & (s_hi > 0)) & (o_lo > 0)  # A0 B+
        l[zp] = s_lo[zp] / o_lo[zp]
        h[zp] = s_hi[zp] / o_lo[zp]
        np = (s_hi <= 0) & (o_lo >= 0)  # A- B+
        l[np] = s_lo[np] / o_lo[np]
        h[np] = s_hi[np] / o_hi[np]
        pn = (s_lo >= 0) & (o_hi <= 0)  # A+ B-
        l[pn] = s_hi[pn] / o_hi[pn]
        h[pn] = s_lo[pn] / o_lo[pn]
        zn = ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0)  # A0 B-
        l[zn] = s_hi[zn] / o_hi[zn]
        h[zn] = s_lo[zn] / o_hi[zn]
        nn = (s_hi <= 0) & (o_hi <= 0)  # A- B-
        l[nn] = s_hi[nn] / o_lo[nn]
        h[nn] = s_lo[nn] / o_hi[nn]
    elif s.scalar:
        l, h = numpy.empty(o_lo.shape), numpy.empty(o_lo.shape)
        pp = (s_lo >= 0) & (o_lo > 0)  # A+ B+
        l[pp] = s_lo / o_hi[pp]
        h[pp] = s_hi / o_lo[pp]
        zp = ((s_lo < 0) & (s_hi > 0)) & (o_lo > 0)  # A0 B+
        l[zp] = s_lo / o_lo[zp]
        h[zp] = s_hi / o_lo[zp]
        np = (s_hi <= 0) & (o_lo >= 0)  # A- B+
        l[np] = s_lo / o_lo[np]
        h[np] = s_hi / o_hi[np]
        pn = (s_lo >= 0) & (o_hi <= 0)  # A+ B-
        l[pn] = s_hi / o_hi[pn]
        h[pn] = s_lo / o_lo[pn]
        zn = ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0)  # A0 B-
        l[zn] = s_hi / o_hi[zn]
        h[zn] = s_lo / o_hi[zn]
        nn = (s_hi <= 0) & (o_hi <= 0)  # A- B-
        l[nn] = s_hi / o_lo[nn]
        h[nn] = s_lo / o_hi[nn]
    elif o.scalar:
        l, h = numpy.empty(s_lo.shape), numpy.empty(s_lo.shape)
        pp = (s_lo >= 0) & (o_lo > 0)  # A+ B+
        l[pp] = s_lo[pp] / o_hi
        h[pp] = s_hi[pp] / o_lo
        zp = ((s_lo < 0) & (s_hi > 0)) & (o_lo > 0)  # A0 B+
        l[zp] = s_lo[zp] / o_lo
        h[zp] = s_hi[zp] / o_lo
        np = (s_hi <= 0) & (o_lo >= 0)  # A- B+
        l[np] = s_lo[np] / o_lo
        h[np] = s_hi[np] / o_hi
        pn = (s_lo >= 0) & (o_hi <= 0)  # A+ B-
        l[pn] = s_hi[pn] / o_hi
        h[pn] = s_lo[pn] / o_lo
        zn = ((s_lo < 0) & (s_hi > 0)) & (o_hi <= 0)  # A0 B-
        l[zn] = s_hi[zn] / o_hi
        h[zn] = s_lo[zn] / o_hi
        nn = (s_hi <= 0) & (o_hi <= 0)  # A- B-
        l[nn] = s_hi[nn] / o_lo
        h[nn] = s_lo[nn] / o_hi
    return l, h
