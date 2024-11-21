""" a Cbox constructor by Leslie"""


from ..pbox_base import Pbox
import numpy as np

__all__ = ['Cbox']

class Cbox(Pbox):
    """
    Confidence boxes (c-boxes) are imprecise generalisations of traditional confidence distributions

    They have a different interpretation to p-boxes but rely on the same underlying mathematics. 
    As such in pba-for-python c-boxes inhert most of their methods from Pbox. 

    Args:
        Pbox (_type_): _description_
    """
    
    def __init__(self,*args,**kwargs):            
        super().__init__(*args,**kwargs)
        print('cbox generated')
        
    
    # def query_confidence(self, level=None, low=None, upper=None):
        
    #     """ or simply the `ci` function 
        
    #     note:
    #         to return the symmetric confidence interval
    #     """
    #     if level is not None:
    #         low = (1-level)/2
    #         upper = 1-low

    #     return self.left(low), self.right(upper)

