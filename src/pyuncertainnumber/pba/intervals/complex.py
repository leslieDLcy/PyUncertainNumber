"""
Created Tue Dec 26 2017
University of Liverpool
github.com/marcodeangelis
MIT License
"""

import numpy
# import random
import matplotlib.pyplot as pyplot

from itertools import repeat

INTERVAL_TYPES=     {'Interval','ComplexInterval','IntervalArray','IntervalDataset'} # for compatibility with other types in this module. Data types sharing same public methods, like lo(), hi(), mid(), etc...

NUMERIC_TYPES =     {'int','float','complex',                   # Python numbers     
                    'int8','int16','int32','int64','intp',      # Numpy integers
                    'uint8','uint16','uint32','uint64','uintp', # Numpy unsigned integers
                    'float16','float32','float64','float_',     # Numpy floats and doubles
                    'complex64','complex128','complex_'}        # Numpy complex floats and doubles

machine_eps = 7./3 - 4./3 - 1


# a = 1+1j
# b = 2+3*1j
# c = -2-1j
# d = 4+5j

class ComplexInterval():
    """
    ∞ --------------------------- ∞
    Created Mon Jul 13 2020
    University of Liverpool
    github.com/marcodeangelis
    MIT License
    ∞ --------------------------- ∞
    """
    def __repr__(self): # return
        return "【%g%+gi, %g%+gi】"%(self.__lo.real,self.__lo.imag,self.__hi.real,self.__hi.imag)

    def __str__(self): # print
        return "【%g%+gi, %g%+gi】"%(self.__lo.real,self.__lo.imag,self.__hi.real,self.__hi.imag)

    def __init__(self,*args):
        self.name = "" # initialise this with an empty string
        self.__subint = 30
        self.__subi_div = False
        self.__subi_mul = False

        if (args is None) | (len(args)==0):
            # raise ValueError('interval constructor needs at least one argument')
            lo, hi = -1-1j, 1+1j
            mid = (lo + hi)/2
            rad = (hi - lo)/2
            width = 2*rad
        if len(args)==1:
            lo, hi = args[0], args[0]
            mid = (lo + hi)/2
            rad = (hi - lo)/2
            width = 2*rad
        if len(args)==2:
            lo, hi = args[0], args[1]
            mid = (lo + hi)/2
            # width = abs(lo.conjugate()*hi)
            rad = (hi - lo)/2
            width = 2*rad


        self.__lo = lo
        self.__hi = hi
        self.__m = mid
        self.__r = rad
        self.__w = width

        self.__iszeroin = [False,False]
        if (lo.real <= 0) & (hi.real >= 0):
            self.__iszeroin[0] = True
        if (lo.imag <= 0) & (hi.imag >= 0):
            self.__iszeroin[1] = True

     # Some handy methods
    def value(self):
        return self
    def inf(self):
        return self.__lo
    def sup(self):
        return self.__hi
    def lo(self):
        return self.__lo
    def hi(self):
        return self.__hi
    def mid(self):
        return self.__m
    def rad(self):
        return self.__r
    def width(self):
        return self.__w
    def stradzero(self):
        return self.__iszeroin
    def slider(self,p):
        return self.__lo + p * self.__w
    def N_subi(self):
        return self.__subint

    def real(self):
        return Interval(self.__lo.real, self.__hi.real)
    def imag(self):
        return Interval(self.__lo.imag, self.__hi.imag)
    def conjugate(self):
        return ComplexInterval(self.__lo.conjugate(),self.__hi.conjugate())
    def absolute(self):
        return (self.real()**2 + self.imag()**2)**0.5

    def set_N_subi(self,N):
        self.__subint = N
        self.__subi_div = True
        return 'Number of subintervals set: division will be performed with %i subintervals'%self.__subint
    def set_subistate(self,state):
        if state:
            self.__subi_div = True
        else:
            self.__subi_div = False
        return r'Subinterval state set to True: division will be performed with %i subintervals'%self.__subint

    def subintervalize(self,*args):
        if len(args)>0:
            N = args[0]
        else:
            N = self.__subint
        def sub1(self,N=N):
            machine_eps = 7./3 - 4./3 - 1
            allowedTypes = ['Interval','I','interval','ComplexInterval']
            if self.__class__.__name__ not in allowedTypes:
                return self
            else:
                if N == 0:
                    out = [self]
                elif N == 1:
                    x_l = Interval(self.inf(),self.mid()+machine_eps)
                    x_r = Interval(self.mid()-machine_eps,self.sup())
                    out = [x_l, x_r]
                else:
                    w = 2*self.rad()/N
                    out = []
                    for n in range(N):
                        x_l = self.inf()+(n+0)*w - machine_eps
                        x_r = self.inf()+(n+1)*w + machine_eps
                        out.append(Interval(x_l,x_r))
            return out
        an = sub1(self.real())
        bn = sub1(self.imag())
        CART =[]
        for ai in an:
            for bi in bn:
                CART.append(ai+bi*1j)
        return CART
    def pop(self,A):
        A_lo = min([ai.lo() for ai in A])
        A_hi = max([ai.hi() for ai in A])
        return Interval(A_lo,A_hi)
    def pop2(self,C):
        Cr_lo = min([ci.real().lo() for ci in C])
        Cr_hi = max([ci.real().hi() for ci in C])
        Ci_lo = min([ci.imag().lo() for ci in C])
        Ci_hi = max([ci.imag().hi() for ci in C])
        return ComplexInterval(Cr_lo+Ci_lo*1j, Cr_hi+Ci_hi*1j)

    #----------------------------------------#
    # Override arithmetical operations START #
    #----------------------------------------#
    def __add__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES: # add support for numpy
            addL = self.__lo + other
            addH = self.__hi + other
            return ComplexInterval(addL,addH)
        elif otherType in INTERVAL_TYPES:
            addL = self.__lo + other.lo()
            addH = self.__hi + other.hi()
            return ComplexInterval(addL,addH)
        else:
            return NotImplemented #print("Error: not among the allowed types.")

    def __radd__(self, left):
        leftType = left.__class__.__name__
        if leftType in NUMERIC_TYPES:
            return self.__add__(left)
        else:
            return NotImplemented #print("Error: not among the allowed types.")
    def __sub__(self, other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            subL = self.__lo - other
            subH = self.__hi - other
            return ComplexInterval(subL,subH)
        elif otherType in INTERVAL_TYPES:
            subL = self.__lo - other.hi()
            subH = self.__hi - other.lo()
            return ComplexInterval(subL,subH)
    def __rsub__(self, left):
        leftType = left.__class__.__name__
        if leftType in NUMERIC_TYPES:
            subL = left - self.__hi
            subH = left - self.__lo
            return ComplexInterval(subL,subH)
        else:
            return NotImplemented #print("Error: not among the allowed types.")


    def __mul__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            # if otherType == 'complex':
            Real = self.real()*other.real - self.imag()*other.imag
            Imag = self.real()*other.imag + self.imag()*other.real
            mulL = Real.lo()+1j*Imag.lo()
            mulH = Real.hi()+1j*Imag.hi()
            return ComplexInterval(mulL,mulH)
        elif otherType in INTERVAL_TYPES:
            if otherType == 'ComplexInterval':
                if self.__subi_mul:
                    OTHER = other.subintervalize()
                    REAL = [self.real()*o.real() - self.imag()*o.imag() for o in OTHER]
                    IMAG = [self.real()*o.imag() + self.imag()*o.real() for o in OTHER]
                    Real = self.pop(REAL)
                    Imag = self.pop(IMAG)
                else:
                    Real = self.real()*other.real() - self.imag()*other.imag()
                    Imag = self.real()*other.imag() + self.imag()*other.real()
                mulL = Real.lo()+1j*Imag.lo()
                mulH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(mulL,mulH)
            else:
                Real = self.real()*other # - self.imag()*other.imag()
                Imag = self.imag()*other #self.real()*other.imag() + self.imag()*other.real()
                mulL = Real.lo()+1j*Imag.lo()
                mulH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(mulL,mulH)
    def __rmul__(self, left):
        leftType = left.__class__.__name__
        if leftType in NUMERIC_TYPES:
            return self.__mul__(left)
        else:
            return NotImplemented
    def __truediv__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            a,b,c,d = self.real(), self.imag(), other.real, other.imag
            Real = (a*c + b*d)/(c**2 + d**2)
            Imag = (b*c - a*d)/(c**2 + d**2)
            divL = Real.lo()+1j*Imag.lo()
            divH = Real.hi()+1j*Imag.hi()
            return ComplexInterval(divL,divH)
        elif otherType in INTERVAL_TYPES:
            if otherType == 'ComplexInterval':
                a,b = self.real(), self.imag()#, other.real(), other.imag()
                if self.__subi_div:
                    OTHER = other.subintervalize()
                    REAL = [(a*o.real() + b*o.imag())/(o.real()**2 + o.imag()**2) for o in OTHER]
                    IMAG = [(b*o.real() - a*o.imag())/(o.real()**2 + o.imag()**2) for o in OTHER]
                    Real = self.pop(REAL)
                    Imag = self.pop(IMAG)
                else:
                    a,b,c,d = self.real(), self.imag(), other.real(), other.imag()
                    Real = (a*c + b*d)/(c**2 + d**2)
                    Imag = (b*c - a*d)/(c**2 + d**2)
                divL = Real.lo()+1j*Imag.lo()
                divH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(divL,divH)
            else:
                a,b,c = self.real(), self.imag(), other
                Real = a/c
                Imag = b/c
                divL = Real.lo()+1j*Imag.lo()
                divH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(divL,divH)
    def __rtruediv__(self, left):
        leftType = left.__class__.__name__
        if leftType in NUMERIC_TYPES:
            if leftType in ['complex']:
                a,b,c,d = left.real, left.imag, self.real(), self.imag()
                Real = (a*c + b*d)/(c**2 + d**2)
                Imag = (b*c - a*d)/(c**2 + d**2)
                divL = Real.lo()+1j*Imag.lo()
                divH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(divL,divH)
            else:
                a,c,d = left, self.real(), self.imag()
                Real = (a*c)/(c**2 + d**2)
                Imag = -(a*d)/(c**2 + d**2)
                divL = Real.lo()+1j*Imag.lo()
                divH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(divL,divH)
        else:
            return NotImplemented
    #------------------------------------------------------------------------------------------------------
    # Override arithmetical operations END
    #------------------------------------------------------------------------------------------------------



class Interval(): 
    '''
    ∞ --------------------------- ∞
    Tue Dec 26 2017
    github.com/marcodeangelis 
    University of Liverpool 
    MIT License
    ∞ --------------------------- ∞
    '''
    def __repr__(self): # return
        return "【%f, %f】"%(self.__lo,self.__hi)  # https://www.compart.com/en/unicode/U+3011 #lenticular brackets

    def __str__(self): # print
        return "【%f, %f】"%(self.__lo,self.__hi)

    def __init__(self,*args):
        if (args is None) | (len(args)==0):
            self.__lo, self.__hi = -1, 1
        if len(args)==1:
            if args[0].__class__.__name__ in NUMERIC_TYPES:
                self.__lo, self.__hi = args[0], args[0]
            elif args[0].__class__.__name__ == 'Interval':
                self = args[0]
            else:
                self.__lo, self.__hi = args[0][0], args[0][1]
        if len(args)==2:
            self.__lo, self.__hi = args[0], args[1]
    def __hash__(self):  # https://docs.python.org/3/reference/datamodel.html#object.__hash__
        return hash((self.lo(),self.hi()))
    ## Class methods start here ##
    def value(self):
        return self
    def lo(self):
        return self.__lo
    def hi(self):
        return self.__hi
    def mid(self):
        return (self.__lo + self.__hi)/2
    def rad(self):
        return (self.__hi - self.__lo)/2
    def halfwidth(self): # Ferson et al. SAND2007-0939
        return (self.__hi - self.__lo)/2
    def width(self):
        return self.__hi - self.__lo
    def diam(self):
        return self.__hi - self.__lo
    def inf(self): #  != lo. support for outword directed rounding
        return self.__lo - machine_eps
    def sup(self): #  != hi. support for outword directed rounding
        return self.__hi + machine_eps
    def mig(self): # mignitude
        return min(abs(self.lo()),abs(self.hi()))
    def mag(self): # magnitude also known as absolute value
        return max(abs(self.lo()),abs(self.hi()))
    def abs(self):
        return self.mag()
    def abs2(self):
        return Interval(self.mig(),self.mag())
    def distance(self,other): # Hausdorff (1972) # Alefeld (1983)
        return max(abs(self.lo()-other.lo()),abs(self.hi()-other.hi()))
    def stradzero(self): # iszeroin
        if (self.__lo <= 0) & (self.__hi >= 0): return True 
        else: return False
    def contains(self,other): # True if self contains other. Works also with non-interval types
        if other.__class__.__name__ not in INTERVAL_TYPES:
            other = Interval(other,other)
        return (self.inf() <= other.inf()) & (self.sup() >= other.sup())
    def encloses(self,other): # True if self encloses other (strictly on both sides)
        return self.lo() < other.lo() and other.hi() < self.hi()
    def inside(self,other): # True if self is inside other
        if other.__class__.__name__ not in INTERVAL_TYPES: # other is a scalar
            other = Interval(other,other)
        return (self.inf() >= other.inf()) & (self.sup() <= other.sup())
    def inside_strict(self,other): # True if self is inside other
        if other.__class__.__name__ not in INTERVAL_TYPES: # other is a scalar
            other = Interval(other,other)
        return (self.inf() > other.inf()) & (self.sup() < other.sup())
    def intersect(self,other): # True if self intersects other and viceversa
        return not(self < other or other < self)
    def union(self,other):
        return Interval(min(self.lo(),other.lo()),max(self.hi(),other.hi()))
    def intersection(self,other):
        if self.intersect(other):
            return Interval(max(self.lo(),other.lo()), min(self.hi(),other.hi()))
        else:
            return None
    def thinit(self,gamma=1,N=1):
        i_m = self.mid()
        i_r = self.halfwidth()
        return Interval(i_m - gamma * i_r/N, i_m + gamma * i_r/N)
    def slider(self,p):
        if p.__class__.__name__ in ['list','tuple']:
            return [self.__lo + pi * self.width() for pi in p]
        else:
            return self.__lo + p * self.width()
    def linspace(self,N=30):
        return list(numpy.linspace(self.lo(),self.hi(),num=N))
    def subintervalize(self,N=30):
        out = None
        if self.__class__.__name__ in INTERVAL_TYPES:
            if N == 0:
                out = [self]
            elif N == 1:
                x_l = Interval(self.inf(),self.mid()+machine_eps)
                x_r = Interval(self.mid()-machine_eps,self.sup())
                out = [x_l, x_r]
            else:
                w = 2*self.rad()/N
                out = []
                for n in range(N):
                    x_l = self.inf()+(n+0)*w - machine_eps
                    x_r = self.inf()+(n+1)*w + machine_eps
                    out.append(Interval(x_l,x_r))
        return out
    def pop(self,A):
        A_lo = min([ai.lo() for ai in A])
        A_hi = max([ai.hi() for ai in A])
        return Interval(A_lo,A_hi)
     ## Class methods end here ##

    #-------------------------------------#
    # Override arithmetic operators START #
    #-------------------------------------#
    # unary operators #
    def __neg__(self):
        return Interval(-self.__hi, -self.__lo)
    def __pos__(self):
        return self
    
    # binary operators #
    def __add__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES: # add support for numpy
            if otherType in ['complex']:
                addL = self.__lo + other
                addH = self.__hi + other
                return ComplexInterval(addL,addH)
            else:
                addL = self.__lo + other
                addH = self.__hi + other
                return Interval(addL,addH)
        elif otherType in INTERVAL_TYPES:
            if otherType == 'ComplexInterval':
                addL = self.__lo + other.lo()
                addH = self.__hi + other.hi()
                return ComplexInterval(addL,addH)
            else:
                addL = self.__lo + other.__lo
                addH = self.__hi + other.__hi
                return Interval(addL,addH)
        else:
            return NotImplemented #print("Error: not among the allowed types.")
    def __radd__(self, left):
        leftType = left.__class__.__name__
        if leftType in NUMERIC_TYPES:
            if leftType in ['complex']:
                addL = self.__lo + left
                addH = self.__hi + left
                return ComplexInterval(addL,addH)
            else:
                addL = left + self.__lo
                addH = left + self.__hi
                return self.__add__(left)
        else:
            return NotImplemented #print("Error: not among the allowed types.")
    def __sub__(self, other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            if otherType in ['complex']:
                subL = self.__lo - other
                subH = self.__hi - other
                return ComplexInterval(subL,subH)
            else:
                subL = self.__lo - other
                subH = self.__hi - other
                return Interval(subL,subH)
        elif otherType in INTERVAL_TYPES:
            if otherType == 'ComplexInterval':
                subL = self.__lo - other.lo()
                subH = self.__hi - other.hi()
                return ComplexInterval(subL,subH)
            else:
                subL = self.__lo - other.__hi
                subH = self.__hi - other.__lo
                return Interval(subL,subH)
    def __rsub__(self, left):
        leftType = left.__class__.__name__
        if leftType in NUMERIC_TYPES:
            if leftType in ['complex']:
                subL = self.__lo - left
                subH = self.__hi - left
                return ComplexInterval(subL,subH)
            else:
                subL = left - self.__hi
                subH = left - self.__lo
                return Interval(subL,subH)
        else:
            return NotImplemented #print("Error: not among the allowed types.")
    def __mul__(self,other):
        otherType = other.__class__.__name__
        if otherType in NUMERIC_TYPES:
            if otherType in ['complex']:
                Real = self*other.real - 0*other.imag
                Imag = self*other.imag + 0*other.real
                mulL = Real.lo()+1j*Imag.lo()
                mulH = Real.hi()+1j*Imag.hi()
                return ComplexInterval(mulL,mulH)
            else:
                if other>0:
                    mulL = self.__lo * other
                    mulH = self.__hi * other
                else:
                    mulL = self.__hi * other
                    mulH = self.__lo * other
        elif otherType in INTERVAL_TYPES:
            if otherType == 'ComplexInterval':
                return other*self # * will be done in ComplexInterval
            else:
                if (self.__lo>=0) & (other.__lo>=0): # A+ B+
                    mulL = self.__lo * other.__lo
                    mulH = self.__hi * other.__hi
                elif (self.__lo>=0) & ((other.__lo<0) & (other.__hi>0)): # A+ B0
                    mulL = self.__hi * other.__lo
                    mulH = self.__hi * other.__hi
                elif (self.__lo>=0) & (other.__hi<=0): # A+ B-
                    mulL = self.__hi * other.__lo
                    mulH = self.__lo * other.__hi
                elif ((self.__lo<0) & (self.__hi>0)) & (other.__lo>=0): # A0 B+
                    mulL = self.__lo * other.__hi
                    mulH = self.__hi * other.__hi
                elif ((self.__lo<0) & (self.__hi>0)) & ((other.__lo<0) & (other.__hi>0)): # A0 B0
                    mulL1 = self.__lo * other.__hi
                    mulL2 = self.__hi * other.__lo
                    mulL = min(mulL1,mulL2)
                    mulH1 = self.__lo * other.__lo
                    mulH2 = self.__hi * other.__hi
                    mulH = max(mulH1,mulH2)
                elif ((self.__lo<0) & (self.__hi>0)) & (other.__hi<=0): # A0 B-
                    mulL = self.__hi * other.__lo
                    mulH = self.__lo * other.__lo
                elif (self.__hi<=0) & (other.__lo>=0): # A- B+
                    mulL = self.__lo * other.__hi
                    mulH = self.__hi * other.__lo
                elif (self.__hi<=0) & ((other.__lo<0) & (other.__hi>0)): # A- B0
                    mulL = self.__lo * other.__hi
                    mulH = self.__lo * other.__lo
                elif (self.__hi<=0) & (other.__hi<=0): # A- B-
                    mulL = self.__hi * other.__hi
                    mulH = self.__lo * other.__lo
        return Interval(mulL,mulH)
    def __rmul__(self, left):
        if left.__class__.__name__ in NUMERIC_TYPES:
            return self.__mul__(left)
        else:
            return NotImplemented
    def __truediv__(self,other):
        if other.__class__.__name__ in NUMERIC_TYPES:
            if other>0:
                divL = self.__lo / other
                divH = self.__hi / other
            elif other<0:
                divL = self.__hi / other
                divH = self.__lo / other
        elif other.__class__.__name__ in INTERVAL_TYPES:
            if other.stradzero():
                raise Warning("Division by interval containing zero") # TODO: extended interval arithmetic
            if (self.__lo>=0) & (other.__lo>0):
                divL = self.__lo/other.__hi
                divH = self.__hi/other.__lo
            elif ((self.__lo<0) & (self.__hi>0)) & (other.__lo>0):
                divL = self.__lo/other.__lo
                divH = self.__hi/other.__lo
            elif (self.__hi<=0) & (other.__lo>0):
                divL = self.__lo/other.__lo
                divH = self.__hi/other.__hi
            elif (self.__lo>=0) & (other.__hi<0):
                divL = self.__hi/other.__hi
                divH = self.__lo/other.__lo
            elif ((self.__lo<0) & (self.__hi>0)) & (other.__hi<0):
                divL = self.__hi/other.__hi
                divH = self.__lo/other.__hi
            elif (self.__hi<=0) & (other.__hi<0):
                divL = self.__hi/other.__lo
                divH = self.__lo/other.__hi
        return Interval(divL,divH)
    def __rtruediv__(self, left):
        if left.__class__.__name__ in NUMERIC_TYPES:
            if left>0:
                if (self.__lo>0):
                    divL = left / self.__hi
                    divH = left / self.__lo
                elif (self.__hi<0):
                    divL = left / self.__hi
                    divH = left / self.__lo
                else:
                    # this should not return an error, but rather an unbounded interval
                    print("Division is allowed for intervals not containing the zero")
                    raise ZeroDivisionError
            elif left<0:
                if (self.__lo>0):
                    divL = left / self.__lo
                    divH = left / self.__hi
                elif (self.__hi<0):
                    divL = left / self.__lo
                    divH = left / self.__hi
                else:
                    # this should not return an error, but rather an unbounded interval
                    print("Division is allowed for intervals not containing the zero")
                    raise ZeroDivisionError
            return Interval(divL,divH)
        else:
            return NotImplemented
    def __pow__(self,other):
        otherType = other.__class__.__name__
        if otherType in INTERVAL_TYPES:
            return NotImplemented #print("Power elevation requires a new operator")
        elif otherType in NUMERIC_TYPES:
            if (other%2==0) | (other%2==1):
                other = int(other)
            if otherType == "int":
                if other > 0:
                    if other%2 == 0: # even power
                        if self.__lo >= 0:
                            powL = self.__lo ** other
                            powH = self.__hi ** other
                        elif self.__hi < 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        else: # interval contains zero
                            H = max(-self.__lo,self.__hi)
                            powL = 0
                            powH = H ** other
                    elif other%2 == 1: # odd power
                        powL = self.__lo ** other
                        powH = self.__hi ** other
                elif other < 0:
                    if other%2 == 0: # even power
                        if self.__lo >= 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        elif self.__hi < 0:
                            powL = self.__lo ** other
                            powH = self.__hi ** other
                        else: # interval contains zero
                            print("Error. \nThe interval contains zero, so negative powers should return \u00B1 Infinity")
                    elif other%2 == 1: # odd power
                        if self.__lo != 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        else: # interval contains zero
                            print("Error. \nThe interval contains zero, so negative powers should return \u00B1 Infinity")
            elif otherType == "float":
                    if self.__lo >= 0:
                        if other > 0:
                            powL = self.__lo ** other
                            powH = self.__hi ** other
                        elif other < 0:
                            powL = self.__hi ** other
                            powH = self.__lo ** other
                        elif other == 0:
                            powL = 1
                            powH = 1
        return Interval(powL,powH)
    def __rpow__(self,left):
        return NotImplemented
    def __lt__(self, other):
        if other.__class__.__name__ in INTERVAL_TYPES:
            return self.sup() < other.inf()
        elif other.__class__.__name__ in NUMERIC_TYPES:
            return self.sup() < other
    def __rlt__(self,left):
        if left.__class__.__name__ in INTERVAL_TYPES:
            return left.sup() < self.inf()
        elif left.__class__.__name__ in NUMERIC_TYPES:
            return left < self.inf()
    def __gt__(self, other):
        if other.__class__.__name__ in INTERVAL_TYPES:
            return self.inf() > other.sup()
        elif other.__class__.__name__ in NUMERIC_TYPES:
            return self.inf() > other
    def __rgt__(self, left):
        if left.__class__.__name__ in INTERVAL_TYPES:
            return left.inf() > self.sup()
        elif left.__class__.__name__ in NUMERIC_TYPES:
            return left > self.sup()
    def __le__(self, other):
        if other.__class__.__name__ in INTERVAL_TYPES:
            return self.sup() <= other.inf()
        elif other.__class__.__name__ in NUMERIC_TYPES:
            return self.sup() <= other
    def __rle__(self,left):
        if left.__class__.__name__ in INTERVAL_TYPES:
            return left.sup() <= self.inf()
        elif left.__class__.__name__ in NUMERIC_TYPES:
            return left <= self.inf()
    def __ge__(self, other):
        if other.__class__.__name__ in INTERVAL_TYPES:
            return self.inf() >= other.sup()
        elif other.__class__.__name__ in NUMERIC_TYPES:
            return self.inf() >= other
    def __rge__(self, left):
        if left.__class__.__name__ in INTERVAL_TYPES:
            return left.inf() >= self.sup()
        elif left.__class__.__name__ in NUMERIC_TYPES:
            return left >= self.sup()
    def __eq__(self, other):
        if other.__class__.__name__ in INTERVAL_TYPES:
            return hash(self)==hash(other)
        else:
            return False
    def __ne__(self,other):
        return not(self == other)
    #     if other.__class__.__name__ in intervalDataTypes():
    #         return (self.sup() == other.sup()) & (self.inf() == other.inf())
    #     elif other.__class__.__name__ in numberDataTypes():
    #         return (self.sup() == other) & (self.inf() == other)
    # def __ne__(self, other):
    #     if other.__class__.__name__ in intervalDataTypes():
    #         return (self.sup() != other.sup()) & (self.inf() != other.inf())
    #     else:
    #         return (self.sup() != other) & (self.inf() != other)
    #------------------------------------------------------------------------------------------------------
    # Override arithmetic operators END
    #------------------------------------------------------------------------------------------------------

def BRUTEFORCE(N=100):  # DECORATOR for "over the top" non-intrusive propagation of intervals
    def decorator(fn):
        def wrapper(*argv):
            u = numpy.random.random_sample((len(argv),N))
            argv_new = []
            for arg,ui in zip(argv,u): # for each variable in the function
                if arg.__class__.__name__ in INTERVAL_TYPES:
                    arg_new = numpy.array(arg.slider(ui))
                else:
                    arg_new = numpy.array([arg]*N)
                argv_new.append(arg_new)
            out = fn(*argv_new)
            if len(out)>1 and out.__class__.__name__ == 'tuple':
                return tuple([Interval(min(o),max(o)) for o in out])
            else:
                return Interval(min(out),max(out))
        return wrapper
    return decorator

def SUBINTERVALIZE(N=20): # DECORATOR for "over the top" subintervalization (intrusive propagation)
    def decorator(fn):
        def wrapper(*argv):
            indices = range(len(argv)) # this will be implementated but not essential at the moment.
            SUBINT_ARGS = []
            MID_ARGS = []
            for i,arg in enumerate(argv): # for each variable in the function
                if (arg.__class__.__name__ in INTERVAL_TYPES) and (i in indices):
                    SUBINT_ARGS.append(arg.subintervalize(N=N))
                    MID_ARGS.append(Interval(arg.mid(),arg.mid()))
                elif arg.__class__.__name__ in INTERVAL_TYPES:
                    SUBINT_ARGS.append(arg)
                    MID_ARGS.append(Interval(arg.mid(),arg.mid()))
                else:
                    SUBINT_ARGS.append(Interval(arg,arg))
                    MID_ARGS.append(Interval(arg,arg))
            OUT_all = fn(*MID_ARGS) # mid point solution used to initiate loop
            d = len(indices) # this is at most D. Indices are needed to reduce the dimensionality of the problem
            for i in range(N**d): # for each cartesian combination
                v = [SUBINT_ARGS[h][i//N**h-(i//N**(h+1))*N] for h in range(d)]
                OUT_s_all = fn(*v)
                if OUT_all.__class__.__name__ == 'tuple':
                    OUT_all = tuple([o.union(s) for o,s in zip(OUT_all,OUT_s_all)])
                else:
                    OUT_all = OUT_all.union(OUT_s_all)
            return OUT_all
        return wrapper
    return decorator

# Wrapper class: 
# (1) simpler spelling,
# (2) memory greedier for CPU intensive tasks (allowing more attributes like name, etc.)
# (3) different notation e.g.: mid-rad
# (4) implement verified version of interval with outward directed rounding
class I(Interval):
    def __init__(self,*args):
        self.name = ''
        super().__init__(*args)
    def superclass(self):
        return self.__class__.__bases__[0].__name__

class interval(Interval):
    def __init__(self,*args):
        self.name = ''
        super().__init__(*args)
    def superclass(self):
        return self.__class__.__bases__[0].__name__


class IntervalArray(): # wrapper class of the scalar class interval with some plotting facilities
    """
    ∞ --------------------------- ∞
    Created Mon Jul 24 2020
    University of Liverpool
    github.com/marcodeangelis
    MIT License
    ∞ --------------------------- ∞
    """
    def __repr__(self): # return
        if len(self)>10:
            a = [str(i) for i in self]
            s = '\n'.join(a[:5]+['...']+a[-5:-1])
        else:
            s = '\n'.join([str(i) for i in self])
        return s
    def __str__(self): # print
        s = '\n'.join([str(i) for i in self])
        return s
    def __len__(self):
        return len([i for i in self])
    def __init__(self,*args,notation='infsup',axis=0,name=''):
        self.name = name # initialise this with an empty string
        if len(args)==0:  # what should an empty IntervalArray(object) look like?
            self.__lo = [-1]
            self.__hi = [1]
        elif len(args)==1:   # this must be a list, tuple (array?) of intervals
            assert args[0].__class__.__name__ in ['list', 'tuple','IntervalArray'], 'single input must be list or a tuple of intervals.'
            if args[0][0].__class__.__name__ in ['Interval','ComplexInterval']:
                self.__lo = [x.lo() for x in args[0]]
                self.__hi = [x.hi() for x in args[0]]
            elif args[0][0].__class__.__name__ in NUMERIC_TYPES:
                self.__lo, self.__hi = args[0], args[0]
            elif args[0][0].__class__.__name__ in ['list','tuple']:
                if axis == 0:
                    if len(args[0])==1:
                        self.__lo, self.__hi = args[0][0], args[0][0]
                    if len(args[0])==2:
                        # assert len(args[0]) == 2, 'a list or tuple is needed.'
                        self.__lo, self.__hi = args[0][0], args[0][1]
                elif axis == 1:
                    self.__lo = list([x for x in zip(*args[0])][0])
                    self.__hi = list([x for x in zip(*args[0])][1])
        elif len(args)==2:
            if args[0].__class__.__name__ == 'list':
                self.__lo, self.__hi = args[0], args[1]
            else:
                self.__lo = list()
                self.__hi = list()
                for a in args:
                    if a.__class__.__name__ == 'tuple':
                        self.__lo.append(a[0]) 
                        self.__hi.append(a[1])
                    elif a.__class__.__name__ == 'Interval':
                        self.__lo.append(a.lo()) 
                        self.__hi.append(a.hi())
                    else:
                        raise TypeError('multiple arguments must be a tuple or an interval.')
    def __iter__(self): # make class iterable
        for l,u in zip(self.__lo,self.__hi):
            yield Interval(l,u)
    def __getitem__(self,index): # make class subscrictable
        if index.__class__.__name__ in ['list','tuple']:
            if len(index)>0:
                return IntervalArray([Interval(self.__lo[i],self.__hi[i]) for i in index])
            else:
                return IntervalArray([]) # todo: create empty dataset
        else:
            return Interval(self.__lo[index],self.__hi[index])
    def inf(self):
        return self.__lo
    def lo(self):
        return self.__lo
    def sup(self):
        return self.__hi
    def hi(self):
        return self.__hi
    def tolist(self):
        return [Interval(l,h) for l,h in zip(self.__lo,self.__hi)]
    def toarray(self, order='F'):
        if order=='F':
            return numpy.array([self.__lo, self.__hi])
        elif order=='C':
            return numpy.array([self.__lo, self.__hi]).T
    def slider(self,p=0.5):  
        if p.__class__.__name__ in ['list','tuple']:
            assert len(self)==len(p), f'p must be of length {len(self)}'
            return [si.slider(pi) for si,pi in zip(self,p)]
        else:
            return [si.slider(p) for si in self] # p = list(repeat(.5, times=len(self)))
    # Magic methods
    def __add__(self,other):
        return IntervalArray([a+b for a,b in zip(self,other)])
    def __sub__(self,other):
        return IntervalArray([a-b for a,b in zip(self,other)])
    def __mul__(self,other):
        return IntervalArray([a*b for a,b in zip(self,other)])
    def __truediv__(self,other):
        return IntervalArray([a/b for a,b in zip(self,other)])
    
    def plot(self,marker='_',size=20,xlabel='x',ylabel='y',title='',save=None):
        N = len(self)
        fig = pyplot.figure(figsize=(18,6))
        ax = fig.subplots()
        x = list(range(0,N))
        ax.plot(x,self.lo())
        ax.plot(x,self.hi())
        ax.fill_between(x=x, y1=self.hi(), y2=self.lo(), alpha=0.3)
        for i in range(0,N):
            ax.scatter([i,i],[self.lo()[i],self.hi()[i]],s=size,marker=marker)
        ax.set_xlabel(xlabel,fontsize=20)
        ax.set_ylabel(ylabel,fontsize=20)
        ax.tick_params(direction='out', length=6, width=2, labelsize=14) #https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
        pyplot.title(title,fontsize=20)
        pyplot.grid()
        if save is not None:
            pyplot.savefig(save)
        pyplot.show()
