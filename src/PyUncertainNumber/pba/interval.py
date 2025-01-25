from typing import Union
import numpy as np
import random as r
import warnings
import matplotlib.pyplot as plt


__all__ = ["Interval", "I", "PM"]


class Interval:
    r"""

    An interval is an uncertain number for which only the endpoints are known, :math:`x=[a,b]`.
    This is interpreted as :math:`x` being between :math:`a` and :math:`b` but with no more information about the value of :math:`x`.

    Intervals embody epistemic uncertainty within PBA.

    Creation
    ________

    Intervals can be created using either of the following:

    .. code-block:: python

        >>> pba.Interval(0,1)
        Interval [0,1]
        >>> pba.I(2,3)
        Interval [2,3]

    .. tip::

        The shorthand ``I`` is an alias for ``Interval``

    Intervals can also be created from a single value ± half-width:

    >>> pba.PM(0,1)
    Interval [-1,1]

    By default intervals are displayed as ``Interval [a,b]`` where ``a`` and ``b`` are the left and right endpoints respectively. This can be changed using the `interval.pm_repr`_ and `interval.lr_repr`_ functions.

    Arithmetic
    __________

    For two intervals [a,b] and [c,d] the following arithmetic operations are defined:

    **Addition**

    :math:`[a,b] + [c,d] = [a+c,b+d]`

    **Subtraction**

    :math:`[a,b] - [c,d] = [a-d,b-c]`

    **Multiplication**

    :math:`[a,b] * [c,d] = [\min(ac,ad,bc,bd),\max(ac,ad,bc,bd)]`

    **Division**

    :math:`[a,b] / [c,d] = [a,b] * \frac{1}{[c,d]} \equiv [\min(a/c,a/d,b/c,b/d),\max(a/c,a/d,b/c,b/d)]`

    Alternative arithmetic methods are described in `interval.add`_, `interval.sub`_, `interval.mul`_, `interval.div`_.

    """

    def __init__(self, left=None, right=None):
        """
        args:
            - left: The left end of the interval.
            - right: The right end of the interval.

        .. admonition:: Default values
        note:
            - If only 1 argument is given then the interval is assumed to be zero width around this value.
            - If no arguments are given then the interval is assumed to be vaccous (i.e. :math:`[-\infty,\infty]`). This is implemented as ``Interval(-np.inf,np.inf)``.
        """

        # disallow p-boxes
        if left.__class__.__name__ == "Pbox" or right.__class__.__name__ == "Pbox":
            raise ValueError(
                "left and right must not be P-boxes. Use Pbox methods instead."
            )

        # assume vaccous if no inputs
        if left is None and right is None:
            right = np.inf
            left = -np.inf

        # If only one input assume zero width
        elif left is None and right is not None:
            left = right
        elif left is not None and right is None:
            right = left

        if hasattr(left, "__iter__") and not isinstance(left, Interval):
            left = Interval(min(left), max(left))
        if hasattr(right, "__iter__") and not isinstance(right, Interval):
            right = Interval(min(right), max(right))

        if isinstance(left, Interval):
            left = left.left
        if isinstance(right, Interval):
            right = right.right

        if left > right:
            LowerUpper = [left, right]
            left = min(LowerUpper)
            right = max(LowerUpper)

        self._left = left
        self._right = right

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    def __repr__(self) -> str:  # return
        return "Interval [%g, %g]" % (self.left, self.right)

    def __str__(self) -> str:  # print
        return "Interval [%g, %g]" % (self.left, self.right)

    def __format__(self, format_spec: str) -> str:
        try:
            return (
                f"[{format(self.left, format_spec)},{format(self.right, format_spec)}]"
            )
        except:
            raise ValueError(
                f"{format_spec} format specifier not understood for Interval object"
            )

    def __iter__(self):
        for bound in [self.left, self.right]:
            yield bound

    def __len__(self):
        return 2

    ###############################
    ##### interval arithmetic #####
    ###############################

    def __radd__(self, left):
        return self.__add__(left)

    def __sub__(self, other):

        if other.__class__.__name__ == "Interval":

            lo = self.left - other.right
            hi = self.right - other.left
        elif other.__class__.__name__ == "Pbox":
            # Perform Pbox subtractnion assuming independance
            return other.rsub(self)
        else:
            try:
                lo = self.left - other
                hi = self.right - other
            except:
                return NotImplemented

        return Interval(lo, hi)

    def __rsub__(self, other):
        if other.__class__.__name__ == "Interval":
            # should be overkill
            lo = other.right - self.left
            hi = other.right - self.right

        elif other.__class__.__name__ == "Pbox":
            # shoud have be caught by Pbox.__sub__()
            return other.__sub__(self)
        else:
            try:
                lo = other - self.right
                hi = other - self.left

            except:
                return NotImplemented

        return Interval(lo, hi)

    def __neg__(self):
        return Interval(-self.right, -self.left)

    def __mul__(self, other):
        if other.__class__.__name__ == "Interval":

            b1 = self.lo * other.lo
            b2 = self.lo * other.hi
            b3 = self.hi * other.lo
            b4 = self.hi * other.hi

            lo = min(b1, b2, b3, b4)
            hi = max(b1, b2, b3, b4)

        elif other.__class__.__name__ == "Pbox":

            return other.mul(self)

        else:

            try:

                lo = self.lo * other
                hi = self.hi * other

            except:

                return NotImplemented

        return Interval(lo, hi)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):

        if other.__class__.__name__ == "Interval":

            if other.straddles_zero():
                if other.left == 0:
                    lo = min(self.lo / other.hi, self.hi / other.hi)
                    hi = np.inf
                elif other.right == 0:
                    lo = -np.inf
                    hi = max(self.lo / other.lo, self.hi / other.lo)
                else:
                    # Cant divide by zero
                    raise ZeroDivisionError()
            else:
                b1 = self.lo / other.lo
                b2 = self.lo / other.hi
                b3 = self.hi / other.lo
                b4 = self.hi / other.hi

                lo = min(b1, b2, b3, b4)
                hi = max(b1, b2, b3, b4)

        elif other.__class__.__name__ == "Pbox":

            return other.__rtruediv__(self)

        else:
            try:
                lo = self.lo / other
                hi = self.hi / other
            except:

                return NotImplemented

        return Interval(lo, hi)

    def __rtruediv__(self, other):

        if self.straddles_zero():

            raise ZeroDivisionError()

        try:
            return other * self.recip()
        except:
            return NotImplemented

    def __pow__(self, other):
        if other.__class__.__name__ == "Interval":
            pow1 = self.left**other.left
            pow2 = self.left**other.right
            pow3 = self.right**other.left
            pow4 = self.right**other.right
            powUp = max(pow1, pow2, pow3, pow4)
            powLow = min(pow1, pow2, pow3, pow4)
        elif other.__class__.__name__ in ("int", "float"):
            pow1 = self.left**other
            pow2 = self.right**other
            powUp = max(pow1, pow2)
            powLow = min(pow1, pow2)
            if (self.right >= 0) and (self.left <= 0) and (other % 2 == 0):
                powLow = 0
        return Interval(powLow, powUp)

    def __rpow__(self, left):
        if left.__class__.__name__ == "Interval":
            pow1 = left.left**self.left
            pow2 = left.left**self.right
            pow3 = left.right**self.left
            pow4 = left.right**self.right
            powUp = max(pow1, pow2, pow3, pow4)
            powLow = min(pow1, pow2, pow3, pow4)

        else:
            pow1 = left**self.left
            pow2 = left**self.right
            powUp = max(pow1, pow2)
            powLow = min(pow1, pow2)

        return Interval(powLow, powUp)

    def __lt__(self, other):
        # <
        if not isinstance(other, Interval):
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"'<' not supported between instances of Interval and {type(other)}"
                )

        if self.right < other.left:
            return Interval(1, 1).to_logical()
        elif self.left > other.right:
            return Interval(0, 0).to_logical()
        else:
            return Interval(0, 1).to_logical()

    def __rgt__(self, other):
        return self < other

    def __eq__(self, other):
        """=="""
        if not isinstance(other, Interval):
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"'==' not supported between instances of Interval and {type(other)}"
                )

        if self.straddles(other.left) or self.straddles(other.right):
            return Interval(0, 1).to_logical()
        else:
            return Interval(0, 0).to_logical()

    def __gt__(self, other):
        """>"""
        try:
            lt = self < other
        except:
            raise TypeError(
                f"'>' not supported between instances of Interval and {type(other)}"
            )
        return ~lt

    def __rlt__(self, other):
        return self > other

    def __ne__(self, other):
        """!="""
        try:
            eq = self == other
        except:
            raise TypeError(
                f"'!=' not supported between instances of Interval and {type(other)}"
            )
        return ~eq

    def __le__(self, other):
        """<="""
        if not isinstance(other, Interval):
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"'<=' not supported between instances of Interval and {type(other)}"
                )

        if self.right <= other.left:
            return Interval(1, 1).to_logical()
        elif self.left > other.right:
            return Interval(0, 0).to_logical()
        else:
            return Interval(0, 1).to_logical()

    def __ge__(self, other):
        if not isinstance(other, Interval):
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"'>=' not supported between instances of Interval and {type(other)}"
                )

        if self.right <= other.left:
            return Interval(1, 1).to_logical()
        elif self.left > other.right:
            return Interval(0, 0).to_logical()
        else:
            return Interval(0, 1).to_logical()

    def __bool__(self):

        try:
            if self.to_logical():
                return True
            else:
                return False
        except:
            raise ValueError("Truth value of Interval %s is ambiguous" % self)

    def __abs__(self):
        if self.straddles_zero():
            return Interval(0, max(abs(self.left), abs(self.right)))
        else:
            return Interval(abs(self.left), abs(self.right))

    def __contains__(self, other):
        return self.straddles(other, endpoints=True)

    def add(self, other, method=None):
        """
        .. _interval.add:

        Adds the interval and another object together.


        **Args**:

            ``other``: The interval or numeric value to be added. This value must be transformable into an Interval object.

        **Methods**:

            p - perfect arithmetic :math:`[a,b]+[c,d] = [a + c, b + d]`

            o - opposite arithmetic :math:`[a,b]+[c,d] = [a + d, b + c]`

            None, i, f - Standard interval arithmetic is used.

        **Returns**:

            ``Interval``

        """
        if not isinstance(other, Interval):
            if other.__class__.__name__ in ["Pbox", "Cbox"]:
                if method is None:
                    method = "f"
                return other.add(self, method=method)
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"addition not supported between instances of Interval and {type(other)}"
                )

        if method == "p":
            Interval(self.left + other.left, self.right + other.right)
        elif method == "o":
            Interval(self.left + other.right, self.right + other.left)
        else:
            return self.__add__(other)

    def __add__(self, other):

        if other.__class__.__name__ == "Interval":
            lo = self.left + other.left
            hi = self.right + other.right
        elif other.__class__.__name__ == "Pbox":
            # Perform Pbox addition assuming independance
            return other.add(self, method="i")
        else:
            try:
                lo = self.left + other
                hi = self.right + other
            except:
                return NotImplemented

        return Interval(lo, hi)

    def padd(self, other):
        """
        .. warning::
            This method is deprecated. Use add(other, method='p') instead.
        """
        warnings.warn(
            "padd() is deprecated. Use add(other, method='p') instead.",
            DeprecationWarning,
        )
        return self.add(other, method="p")

    def oadd(self, other):
        """
        .. warning::
            This method is deprecated. Use add(other, method='o') instead.
        """
        warnings.warn(
            "oadd() is deprecated. Use add(other, method = 'o') instead.",
            DeprecationWarning,
        )
        return self.add(other, method="o")

    def sub(self, other, method=None):
        """
        .. _interval.sub:

        Subtracts other from self.

        **Args**:

            ``other``: The interval or numeric value to be subracted. This value must be transformable into an Interval object.

        **Methods**:

            ``p``: perfect arithmetic :math:`a+b = [a.left - b.left, a.right - b.right]`

            ``o``: opposite arithmetic :math:`a+b = [a.left - b.right, a.right - b.left]`

            None, i, f - Standard interval arithmetic is used.

        **Returns**:

            ``Interval``

        """
        if not isinstance(other, Interval):
            if other.__class__.__name__ in ["Pbox", "Cbox"]:
                if method is None:
                    method = "f"
                return other.rsub(self, method=method)
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"Subtraction not supported between instances of Interval and {type(other)}"
                )

        if method == "p":
            Interval(self.left - other.left, self.right - other.right)
        elif method == "o":
            Interval(self.left - other.right, self.right - other.left)
        else:
            return self.__sub__(other)

    def psub(self, other):
        """
        .. warning::
            Depreciated use self.sub(other, method = 'p') instead
        """
        warnings.warn(
            "psub() is deprecated. Use sub(other, method = 'p') instead.",
            DeprecationWarning,
        )
        return Interval(self.left - other.left, self.right - other.right)

    def osub(self, other):
        """
        .. warning::
            Depreciated use self.sub(other, method = 'o') instead
        """
        warnings.warn(
            "osub() is deprecated. Use sub(other, method = 'o') instead.",
            DeprecationWarning,
        )
        return Interval(self.left - other.right, self.right - other.left)

    def mul(self, other, method=None):
        """
        .. _interval.mul:

        Multiplies self by other.

        **Args**:

            ``other``: The interval or numeric value to be multiplied. This value must be transformable into an Interval object.


        **Methods**:
            ``p``: perfect arithmetic :math:`[a,b],[c,d] = [a * c, b * d]`

            ``o``: opposite arithmetic :math:`[a,b],[c,d] = [a * d, b * c]`

            None, i, f - Standard interval arithmetic is used.

        **Returns**:
            Interval: The result of the multiplication.


        """
        if not isinstance(other, Interval):
            if other.__class__.__name__ in ["Pbox", "Cbox"]:
                if method is None:
                    method = "f"
                return other.mul(self, method=method)
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"Multiplication not supported between instances of Interval and {type(other)}"
                )

        if method == "p":
            return Interval(self.left * other.left, self.right * other.right)
        elif method == "o":
            return Interval(self.left * other.right, self.right * other.left)
        else:
            return self.__mul__(other)

    def pmul(self, other):
        """
        .. warning::
            Depreciated use self.mul(other, method = 'p') instead
        """
        warnings.warn(
            "pmul() is deprecated. Use mul(other, method = 'p') instead.",
            DeprecationWarning,
        )
        return Interval(self.left * other.left, self.right * other.right)

    def omul(self, other):
        """
        .. warning::
            Depreciated use self.mul(other, method = 'o') instead
        """
        warnings.warn(
            "omul() is deprecated. Use mul(other, method = 'o') instead.",
            DeprecationWarning,
        )
        return Interval(self.left * other.right, self.right * other.left)

    def div(self, other, method=None):
        """
        .. _interval.div:

        Divides self by other


        If :math:`0 \\in other` it returns a division by zero error


        **Args**:

            ``other`` (Interval or numeric): The interval or numeric value to be multiplied. This value must be transformable into an Interval object.

        **Methods**:

            ``p``: perfect arithmetic :math:`[a,b],[c,d] = [a * 1/c, b * 1/d]`

            ``o``: opposite arithmetic :math:`[a,b],[c,d] = [a * 1/d, b * 1/c]`

            ``None``, ``i``, ``f`` - Standard interval arithmetic is used.

        .. admonition:: Implementation

            >>> self.add(1/other, method = method)

        .. error::

            If :math:`0 \\in [a,b]` it returns a division by zero error

        """
        if not isinstance(other, Interval):
            if other.__class__.__name__ in ["Pbox", "Cbox"]:
                if method is None:
                    return other.recip().mul(self, method="f")
                elif method == "o":
                    return other.recip().mul(self, method="p")
                elif method == "p":
                    return other.recip().mul(self, method="o")
                else:
                    return other.recip().mul(self, method=method)
            try:
                other = Interval(other)
            except:
                raise TypeError(
                    f"Division not supported between instances of Interval and {type(other)}"
                )

        if method == "o":
            return Interval(self.left / other.right, self.right / other.left)
        elif method == "p":
            return Interval(self.left / other.left, self.right / other.right)

        return self.mul(1 / other, method=method)

    def pdiv(self, other):
        """
        .. warning::
            Depreciated use self.div(other, method = 'p') instead
        """
        warnings.warn(
            "pdiv() is deprecated. Use div(other, method = 'p') instead.",
            DeprecationWarning,
        )
        return Interval(self.left / other.left, self.right / other.right)

    def odiv(self, other):
        """
        .. warning::
            Depreciated use self.div(other, method = 'o') instead

        """
        return Interval(self.left / other.right, self.right / other.left)

    def recip(self):
        """
        Calculates the reciprocle of the interval.

        **Returns**:

            ``Interval``: Equal to :math:`[1/b,1/a]`

        **Example**:

            >>> pba.Interval(2,4).recip()
            Interval [0.25, 0.5]

        .. error::
            If :math:`0 \\in [a,b]` it returns a division by zero error

        """
        if self.straddles_zero():
            # Cant divide by zero
            raise ZeroDivisionError()

        elif 1 / self.hi < 1 / self.lo:
            return Interval(1 / self.hi, 1 / self.lo)
        else:
            return Interval(1 / self.lo, 1 / self.hi)

    def equiv(self, other: "Interval") -> bool:
        """
        Checks whether two intervals are equivalent.

        **Parameters**:

            ``other``: The interval to check against.

        **Returns** ``True`` **if**:

            ``self.left == other.right`` and ``self.right == other.right``

            ``False`` otherwise.

        .. error::

            ``TypeError``: If ``other`` is not an instance of ``Interval``

        .. seealso::
            :func:`~logical.is_same_as`

        **Examples**:

            >>> a = Interval(0,1)
            >>> b = Interval(0.5,1.5)
            >>> c = I(0,1)
            >>> a.equiv(b)
            False
            >>> a.equiv(c)
            True

        """
        if not isinstance(other, Interval):
            raise TypeError(f"Needs to be an instance of Interval not {type(other)}")

        return self.left == other.left and self.right == other.right

    @property
    def lo(self):
        """
        **Returns**:
            ``self.left``

        .. tip::

            This function is redundant but exists to match Pbox class for possible internal reasons.

        """
        return self._left

    @property
    def hi(self):
        """
        **Returns**:
            ``self.right``

        .. tip::

            This function is redundant but exists to match Pbox class for possible internal reasons.

        """
        return self._right

    def width(self) -> float:
        """
        **Returns**:

            ``float``: The width of the interval, :math:`\mathrm{right} - \mathrm{left}`

        **Example**:

            >>> pba.Interval(0,3).width()
            3

        """
        return self.right - self.left

    def halfwidth(self) -> float:
        """
        **Returns**:

            ``float``: The half-width of the interval, :math:`(\mathrm{right} - \mathrm{left})/2`

        **Example**:

            >>> pba.Interval(0,3).halfwidth()
            1.5

        .. admonition:: Implementation

            >>> self.width()/2

        """
        return self.width() / 2

    def midpoint(self) -> float:
        """Returns the midpoint of the interval

        note:
            - this serves as the deterministic value representation of the interval, a.k.a. the naked value for an interval

        returns:
            ``float``: The midpoint of the interval, :math:`(\mathrm{right} + \mathrm{left})/2`

        **Example**:
                >>> pba.Interval(0,2).midpoint()
                1.0
        """

        return (self.left + self.right) / 2

    def to_logical(self):
        """
        Turns the interval into a logical interval, this is done by chacking the truth value of the ends of the interval

        **Returns**:

            ``Logical``: The logical interval

        .. admonition:: Implementation

            >>> left = self.left.__bool__()
            >>> right = self.right.__bool__()
            >>> Logical(left,right)


        """

        from .logical import Logical

        return Logical(self.left.__bool__(), self.right.__bool__())

    def env(self, other: Union[list, "Interval"]) -> "Interval":
        """Calculates the envelope between two intervals

        **Parameters**:

            ``other`` : Interval or list. The interval to envelope with self

        .. hint::

            If other is a list then the envelope is calculated between self and each element of the list. In this case the envelope is calculated recursively and pba.envelope() may be more efficient.

        .. important::
            If other is a Pbox then ``Pbox.env()`` is called

        .. seealso::

            `pba.core.envelope`_

            `pba.pbox.Pbox.env`_

        **Returns**:

            ``Interval``: The envelope of self and other


        """
        if isinstance(other, [list, tuple]):
            e = self
            for o in other:
                e = e.env(o)
            return e

        if not isinstance(other, Interval):
            if other.__class__.__name__ in ["Pbox", "Cbox"]:
                return other.env(self)
            else:
                try:
                    other = Interval(other)
                except:
                    raise TypeError(
                        f"env() not supported between instances of Interval and {type(other)}"
                    )

        return Interval(min(self.left, other.left), max(self.right, other.right))

    def straddles(
        self, N: Union[int, float, "Interval"], endpoints: bool = True
    ) -> bool:
        """
        .. _interval.straddles:

        **Parameters**:

            ``N``: Number to check. If N is an interval checks whether the whole interval is within self.

            ``endpoints``: Whether to include the endpoints within the check

        **Returns** ``True`` **if**:

            :math:`\mathrm{left} \leq N \leq \mathrm{right}` (Assuming ``endpoints=True``).

            For interval values. :math:`\mathrm{left} \leq N.left \leq \mathrm{right}` and :math:`\mathrm{left} \leq N.right \leq \mathrm{right}` (Assuming ``endpoints=True``).

            ``False`` otherwise.

        .. tip::

            ``N in self`` is equivalent to ``self.straddles(N)``

        """
        if isinstance(N, Interval):
            return self.straddles(N.left, endpoints) and self.straddles(
                N.right, endpoints
            )

        if endpoints:
            if self.left <= N and self.right >= N:
                return True
        else:
            if self.left < N and self.right > N:
                return True

        return False

    def straddles_zero(self, endpoints=True):
        """
        Checks whether :math:`0` is within the interval

        .. admonition:: Implementation

            Equivalent to ``self.straddles(0,endpoints)``

        .. seealso::
            interval.straddles_

        """
        return self.straddles(0, endpoints)

    def intersection(self, other: Union["Interval", list]) -> "Interval":
        """Calculates the intersection between intervals

        **Parameters**:

            ``other``: The interval to intersect with self. If an interval is not given will try to cast as an interval. If a list is given will calculate the intersection between self and each element of the list.

        **Returns**:

            ``Interval``: The intersection of self and other. If no intersection is found returns ``None``

        **Example**:

            >>> a = Interval(0,1)
            >>> b = Interval(0.5,1.5)
            >>> a.intersection(b)
            Interval [0.5, 1]
        """
        if isinstance(other, Interval):
            if self.straddles(other):
                return I(
                    max([x.left for x in [self, other]]),
                    min([x.right for x in [self, other]]),
                )
            else:
                return None
        elif isinstance(other, list):
            if all([self.straddles(o) for o in other]):
                assert all(
                    [isinstance(o, Interval) for o in other]
                ), "All intersected objects must be intervals"
                return I(
                    max([x.left for x in [self] + other]),
                    min([x.right for x in [self] + other]),
                )
            else:
                return None
        else:
            if self.straddles(other):
                return other
            else:
                return None

    def exp(self):
        lo = np.exp(self.left)
        hi = np.exp(self.right)
        return Interval(lo, hi)

    def log(self):
        lo = np.log(self.left)
        hi = np.log(self.right)
        return Interval(lo, hi)

    def sqrt(self):
        if self.left >= 0:
            return Interval(np.sqrt(self.left), np.sqrt(self.right))
        else:
            print("RuntimeWarning: invalid value encountered in sqrt")
            return Interval(np.nan, np.sqrt(self.right))

    # def sin(self):
    #     return Interval(np.sin(self.left),np.sin(self.right))
    # def cos(self):
    #     return Interval(np.cos(self.left),np.cos(self.right))
    # def tan(self):
    #     return Interval(np.tan(self.left),np.tan(self.right))

    # ---------------------methods---------------------#

    def display(self, title="", ax=None, style="band", **kwargs):

        if ax is None:
            fig, ax = plt.subplots()

        if title is not None:
            ax.set_title(title, **kwargs)

        if style == "simple":
            # Plot the interval horizontally
            ax.plot(
                [self.left, self.right],
                [
                    1,
                    1,
                ],
                "blue",
                **kwargs,
            )
        elif style == "band":
            # plot the band
            ax.fill_between(
                x=[self.left, self.right],
                y1=0,
                y2=1,
                edgecolor="blue",
                facecolor=("blue", 0.2),
                **kwargs,
            )
        else:
            raise ValueError(f"Style {style} not recognised")
        ax.margins(x=0.1)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel("CDF")
        return ax

    def sample(self, seed=None, numpy_rng: np.random.Generator = None) -> float:
        """
        Generate a random sample within the interval.

        **Parameters**:

            ``seed`` (int, optional): Seed value for random number generation. Defaults to None.

            ``numpy_rng`` (numpy.random.Generator, optional): Numpy random number generator. Defaults to None.

        **Returns**:

            ``float``: Random sample within the interval.

        .. admonition:: Implementation

            If ``numpy_rng`` is given:

                >>> numpy_rng.uniform(self.left, self.right)

            Otherwise the following is used:

            >>> import random
            >>> random.seed(seed)
            >>> self.left + random.random() * self.width()

        **Examples**:

            >>> pba.Interval(0,1).sample()
            0.6160988752201705
            >>> pba.I(0,1).sample(seed = 1)
            0.13436424411240122

        If a numpy random number generator is given then it is used instead of the default python random number generator. It has to be initialised first.

            >>> import numpy as np
            >>> rng = np.random.default_rng(seed = 0)
            >>> pba.I(0,1).sample(numpy_rng = rng)
            0.6369616873214543
        """

        if numpy_rng is not None:
            return numpy_rng.uniform(self.left, self.right)

        if seed is not None:
            r.seed(seed)
        return self.left + r.random() * self.width()

    def round(
        self,
    ):
        """outward rounding operation for an interval object"""
        return Interval(np.floor(self.left), np.ceil(self.right))

    # ---------------------Constructors---------------------#

    @classmethod
    def from_midwith(cls, midpoint: float, halfwidth: float) -> "Interval":
        """Creates an Interval object from a midpoint and half-width.

        args:
            - midpoint (float): The midpoint of the interval.
            - halfwidth (float): The half-width of the interval.

        returns:
            - Interval: The interval with midpoint and half-width.

        example:
            >>> pba.Interval.from_midwith(0,1)
            Interval [-1, 1]
        """

        return cls(midpoint - halfwidth, midpoint + halfwidth)


# Alias
I = Interval


def PM(x, hw):
    """Create an interval centered around x with a half-width of hw.

    # TODO: a constructor for creating an interval from a midpoint and half-width
    # ! this func is weird and I keep it only for not breaking the code in other places.

    **Parameters**:

        ``x`` (float): The center value of the interval.

        ``hw`` (float): The half-width of the interval.

    **Returns**:

        ``Interval``: An interval object with lower bound x-hw and upper bound x+hw.

    .. error::

        ``ValueError``: If hw is less than 0.

    Example:
        >>> pba.PM(0, 1)
        Interval [-1, 1]

    """
    if hw < 0:
        raise ValueError("hw must be greater than or equal to 0")

    return Interval(x - hw, x + hw)


def pm_repr():
    """
    .. _interval.pm_repr:

    Modifies the interval class to display the interval in [midpoint ± half-width] format.

    **Example**:

    .. code-block:: python

        >>> import pba
        >>> pba.interval.pm_repr()
        >>> a = pba.Interval(0,1) # defined using left and right. This cannot be overriden.
        >>> b = pba.PM(0,1) # defined using midpoint and half-width
        >>> print(a)
        Interval [0.5 ± 0.5]
        >>> print(b)
        Interval [0 ± 1]

    .. seealso::
        :func:`~pba.lr_interval_repr`

    """

    def new__repr__(self):
        return f"Interval [{self.midpoint():g} ± {self.halfwidth():g}]"

    Interval.__repr__ = new__repr__
    Interval.__str__ = new__repr__


def lr_repr():
    """
    .. _interval.lr_repr:

    Modifies the interval class to display the interval in [left, right] format.

    .. note::
        This function primarily exists to undo the effects of ``pm_interval_repr()``. By default the interval class displays in this format.

    **Example**:

    .. code-block:: python

        >>> import pba
        >>> pba.interval.pm_repr()
        >>> a = pba.Interval(0,1) # defined using left and right, this cannot be overriden.
        >>> print(a)
        Interval [0.5±0.5]
        >>> pba.interval.lr_repr()
        >>> b = pba.PM(0,1) # defined using midpoint and half-width
        >>> print(b)
        Interval [-1,1]

    """

    def new__repr__(self):
        return f"Interval [{self.left:g}, {self.right:g}]"

    Interval.__repr__ = new__repr__
    Interval.__str__ = new__repr__
