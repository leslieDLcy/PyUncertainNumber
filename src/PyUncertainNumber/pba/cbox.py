if __name__ is not None and "." in __name__:
    from .pbox import Pbox
else:
    from pbox import Pbox
    
import numpy as np

__all__ = ['Cbox']

class Cbox(Pbox):
    """
    Confidence boxes (c-boxes) are imprecise generalisations of traditional confidence distributions

    They have a different interpretation to p-boxes but rely on the same underlying mathematics. As such in pba-for-python c-boxes inhert most of their methods from Pbox. 

    Args:
        Pbox (_type_): _description_
    """
    
    def __init__(self,*args,**kwargs):
        if len(args) == 1 and isinstance(args[0],Pbox):
            
                super().__init__(**vars(args[0]))
                
        else:
            
            super().__init__(*args,**kwargs)
        
    def __repr__(self):
        if self.mean_left == self.mean_right:
            mean_text = f'{round(self.mean_left, 4)}'
        else:
            mean_text = f'[{round(self.mean_left, 4)}, {round(self.mean_right, 4)}]'

        if self.var_left == self.var_right:
            var_text = f'{round(self.var_left, 4)}'
        else:
            var_text = f'[{round(self.var_left, 4)}, {round(self.var_right, 4)}]'

        range_text = f'[{round(np.min([self.left, self.right]), 4), round(np.max([self.left, self.right]), 4)}'

        if self.shape is None:
            shape_text = ' '
        else:
            shape_text = f' {self.shape}' # space to start; see below lacking space

        return f'Cbox: ~ {shape_text} (range={range_text}, mean={mean_text}, var={var_text})'

    __str__ = __repr__

    def __add__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__add__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__add__(other))

    def __radd__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__radd__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__radd__(other))

    def add(self,other, method = 'f'):
        return Cbox(super().add(other, method = method))

    def __sub__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__sub__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__sub__(other))

    def __rsub__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__rsub__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__rsub__(other))

    def sub(self,other, method = 'f'):
        return Cbox(super().sub(other, method = method))

    def __mul__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__mul__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__mul__(other))

    def __rmul__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__rmul__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__rmul__(other))

    def mul(self,other, method = 'f'):
        return Cbox(super().mul(other, method = method))

    def __truediv__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__truediv__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__truediv__(other))

    def __rtruediv__(self, other):
        if isinstance(other, Cbox):
            return Cbox(super().__rtruediv__(other))
        elif isinstance(other, Pbox):
            raise NotImplementedError
        else:
            return Cbox(super().__rtruediv__(other))

    def div(self,other, method = 'f'):
        return Cbox(super().div(other, method = method))

    def __neg__(self):
        return Cbox(super().__neg__())
    
    def recip(self):
        return Cbox(super().recip())
    
