from functools import wraps


class NominalValueMixin:
    @property
    def nominal_value(self):
        return self._compute_nominal_value()

    def _compute_nominal_value(self):
        raise NotImplementedError


# --- mixin that forwards ops to Pbox after converting self/other ---
class _PboxOpsMixin:
    """
    Mixin for classes that want to reuse Pbox's arithmetic by
    converting themselves (and the other operand) into Pbox first.
    Requires a _to_pbox(self) -> Pbox method on the subclass.
    """

    # if you also want comparisons, add them similarly
    _BIN_OPS = [
        "__add__",
        "__sub__",
        "__mul__",
        "__truediv__",
        "__floordiv__",
        "__mod__",
        "__pow__",
        "__matmul__",
        "__and__",
        "__or__",
        "__xor__",
        "__lshift__",
        "__rshift__",
    ]
    _REFL = {  # reflected names
        "__add__": "__radd__",
        "__sub__": "__rsub__",
        "__mul__": "__rmul__",
        "__truediv__": "__rtruediv__",
        "__floordiv__": "__rfloordiv__",
        "__mod__": "__rmod__",
        "__pow__": "__rpow__",
        "__matmul__": "__rmatmul__",
        "__and__": "__rand__",
        "__or__": "__ror__",
        "__xor__": "__rxor__",
        "__lshift__": "__rlshift__",
        "__rshift__": "__rrshift__",
    }
    _UNARY_OPS = ["__neg__", "__pos__", "__abs__", "__invert__"]

    @staticmethod
    def _coerce_to_pbox(x):
        from .pbox_abc import Pbox

        if isinstance(x, Pbox):
            return x
        # If the other operand is a DSS (or another _PboxOpsMixin), convert it too
        if isinstance(x, _PboxOpsMixin):
            return x._to_pbox()
        return x  # let Pbox handle plain numbers, numpy arrays, etc.

    # install methods on the class at definition time
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        from .pbox_abc import Pbox

        # binary ops
        for name in cls._BIN_OPS:
            refl = cls._REFL[name]

            def make_bin(nm):
                @wraps(getattr(Pbox, nm, lambda *a, **k: None))
                def _op(self, other):
                    a = self._to_pbox()
                    b = cls._coerce_to_pbox(other)
                    meth = getattr(a, nm, None)
                    if meth is None:
                        return NotImplemented
                    return meth(b)

                return _op

            def make_rbin(rnm):
                @wraps(getattr(Pbox, rnm, lambda *a, **k: None))
                def _rop(self, other):
                    a = cls._coerce_to_pbox(other)
                    b = self._to_pbox()
                    meth = getattr(b, rnm, None)
                    if meth is None:
                        return NotImplemented
                    return meth(a)

                return _rop

            setattr(cls, name, make_bin(name))
            setattr(cls, refl, make_rbin(refl))

        # unary ops
        for name in cls._UNARY_OPS:

            def make_unary(nm):
                @wraps(getattr(Pbox, nm, lambda *a, **k: None))
                def _u(self):
                    return getattr(self._to_pbox(), nm)()

                return _u

            setattr(cls, name, make_unary(name))

    # Optional: delegate *other* methods/attributes to the Pbox view
    def __getattr__(self, name):
        # Called only if normal lookup fails, so avoid shadowing real attrs
        return getattr(self._to_pbox(), name)
