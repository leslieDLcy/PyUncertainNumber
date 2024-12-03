""" a Cbox constructor by Leslie"""
from .params import Params
from .pbox_base import Pbox
from ..UC.utils import tranform_ecdf


class Cbox(Pbox):
    """
    Confidence boxes (c-boxes) are imprecise generalisations of traditional confidence distributions

    They have a different interpretation to p-boxes but rely on the same underlying mathematics. 
    As such in pba-for-python c-boxes inhert most of their methods from Pbox. 

    Args:
        Pbox (_type_): _description_
    """

    def __init__(self, *args, extre_bound_params=None, **kwargs):
        """ Cbox constructor

        args:
            extre_bound_params: envelope (extreme) bounds of the box
        """
        self.extre_bound_params = extre_bound_params
        super().__init__(*args, **kwargs)

    def __repr__(self):
        # msg =
        return f"Cbox ~ {self.shape}{self.extre_bound_params}"

    def display(self, parameter_name=None, **kwargs):
        ax = super().display(
            title=f'Cbox {parameter_name}', fill_color='salmon', **kwargs)
        ax.set_ylabel('Confidence')
        return ax

    @classmethod
    def from_cd(cls):
        """ constructor from a confidence distribution """

        pass

    # def query_confidence(self, level=None, low=None, upper=None):

    #     """ or simply the `ci` function

    #     note:
    #         to return the symmetric confidence interval
    #     """
    #     if level is not None:
    #         low = (1-level)/2
    #         upper = 1-low

    #     return self.left(low), self.right(upper)

# * ---------------------  constructors--------------------- *#


def cbox_from_envdists(rvs, shape=None, extre_bound_params=None):
    """ define cbox via extreme bouding distrbution functions

    args:
        rvs (list): list of `scipy.stats.rv_continuous` objects
    """
    if not isinstance(rvs, list | tuple):
        rvs = [rvs]
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    # if extre_bound_params is not None: print(extre_bound_params)

    return Cbox(
        *bounds,
        extre_bound_params=extre_bound_params,
        shape=shape,
    )

# used for nextvalue distribution which by nature is pbox


def cbox_from_pseudosamples(samples):

    return Cbox(tranform_ecdf(samples, display=False))


# * ---------------------next value --------------------- *#

def repre_pbox(rvs, shape="beta", extre_bound_params=None):
    """ transform into pbox object for cbox 

    args:
        rvs (list): list of scipy.stats.rv_continuous objects"""

    # x_sup
    bounds = [rv.ppf(Params.p_values) for rv in rvs]
    if extre_bound_params is not None:
        print(extre_bound_params)
    return Pbox(
        left=bounds[0],
        right=bounds[1],
        shape=shape,
    )


def pbox_from_pseudosamples(samples):
    """ a tmp constructor for pbox/cbox from approximate solution of the confidence/next value distribution 

    args:
        samples: the approximate Monte Carlo samples of the confidence/next value distribution

    note:
        ecdf is estimted from the samples and bridge to pbox/cbox
    """
    return Pbox(tranform_ecdf(samples, display=False))
