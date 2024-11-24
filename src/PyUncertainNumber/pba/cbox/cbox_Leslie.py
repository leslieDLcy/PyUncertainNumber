""" a Cbox constructor by Leslie"""


from ..pbox_base import Pbox

__all__ = ['Cbox']

class Cbox(Pbox):
    """
    Confidence boxes (c-boxes) are imprecise generalisations of traditional confidence distributions

    They have a different interpretation to p-boxes but rely on the same underlying mathematics. 
    As such in pba-for-python c-boxes inhert most of their methods from Pbox. 

    Args:
        Pbox (_type_): _description_
    """
    
    def __init__(self, *args, bound_params=None, **kwargs):            
        self.bound_params = bound_params
        super().__init__(*args,**kwargs)


    def __repr__(self):
        # msg = 
        return f"Cbox ~ {self.shape}{self.bound_params}"
    

    def display(self, parameter_name=None, **kwargs):
        ax = super().display(title=f'Cbox {parameter_name}', fill_color='salmon', **kwargs)
        ax.set_ylabel('Confidence')
        return ax


    # def query_confidence(self, level=None, low=None, upper=None):
        
    #     """ or simply the `ci` function 
        
    #     note:
    #         to return the symmetric confidence interval
    #     """
    #     if level is not None:
    #         low = (1-level)/2
    #         upper = 1-low

    #     return self.left(low), self.right(upper)

