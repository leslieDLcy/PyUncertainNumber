import numpy as np

from pyuncertainnumber import pba
from pyuncertainnumber.pba.pbox_abc import Staircase
from ..pba.dependency import Dependency
from ..pba.distributions import JointDistribution

"""independence of input pboxes are assumed to get started"""


def middle_pinch(pbox_list: list):
    """Perform middle pinch on a list of p-boxes.

    args:
        pbox_list (list): A list of marginal p-boxes to be processed.
    """
    from pyuncertainnumber.pba.pbox_abc import Staircase

    def middle_pinch_function(pbox):
        q = (pbox.left + pbox.right) / 2
        return Staircase(left=q, right=q)

    return [middle_pinch_function(pbox) for pbox in pbox_list]


class SurrogatePropagation:
    """Surrogate propagation for uncertain numbers: (Pbox for now)

    args:

        vars (uncertain number or constructs):
            input pboxes

        surrogate_model:
            supported surrogate models in `probabilistic_model_archive`, {IPM, GP, ...}

        N1 (int): number of experimental design for level-1 meta-modelling

        N2 (int): number of experimental design for level-2 meta-modelling

        f (callable): True DGM computational model;

    note:
        - notable examples: GP or IPM to propagate p-boxes for imprecise reliability analysis;
        - Independence of input p-boxes is assumed to get started.
        - the two-level framework is used.
    """

    def __init__(
        self,
        vars,
        f,
        surrogate_model,
        N1=100,
        N2=200,
        dependency: Dependency = None,
        auxiliary_input_distribution=None,
        random_state=None,
    ):
        self.vars = vars
        self.f = f
        self.surrogate_model = surrogate_model
        self.N1 = N1
        self.N2 = N2
        self.dependency = dependency
        self.auxiliary_input_distribution = (
            middle_pinch(self.vars)
            if auxiliary_input_distribution is None
            else auxiliary_input_distribution
        )
        self.random_state = random_state
        ### add the level-1 and level-2 logic herein
        # self.level_1_meta_modelling()
        # self.level_2_meta_modelling()
        # self.use_l2_get_response()

    def level_1_meta_modelling(self):
        # Implement level-1 meta-modelling here
        from pyuncertainnumber.pba.intervals.number import interval_degenerate

        alpha_l1 = self.dependency.u_sample(
            self.N1, random_state=self.random_state
        )  # (N1, d)

        X_tilda_input = []
        for i, v in enumerate(self.auxiliary_input_distribution):
            arr = interval_degenerate(v.alpha_cut(alpha_l1[:, i]))
            X_tilda_input.append(arr)
        X_tilda_input = np.array(X_tilda_input).T  # (N1, d)

        y_tilda_output = self.f(X_tilda_input)  # get real y_obs

        return X_tilda_input, y_tilda_output

        ### to train a level-1 surrogate model ###
        # self.l1_model = self.surrogate_model.fit(X_tilda_input, y_tilda_output)

    def level_2_meta_modelling(self, trained_surrogate):
        """Level-2 meta-modelling.

        This step involves interval propagation accomplished by optimisation, therefore it takes some time to execute.

        note:
            mapping between ɑ and y_low, ɑ and y_high. an IPM model can achieve the mapping using just one model.
        """
        from pyuncertainnumber import b2b

        ndim = len(self.vars)
        prob_proxy_input = self.dependency.u_sample(
            self.N2, random_state=self.random_state
        )  # (N2, d)
        y_lo = np.zeros(len(prob_proxy_input))
        y_hi = np.zeros(len(prob_proxy_input))

        for i, row in enumerate(prob_proxy_input):
            x_domain = [
                v.alpha_cut(a) for v, a in zip(self.vars, row)
            ]  # yield a list of intervals
            # TODO: fix the issue of being too slow when using surrogate model for optimisation
            response_y_itvl = b2b(
                vars=x_domain,
                func=self.f,  # temporarily use true DGM to replace surrogate
                # func=trained_surrogate,
                interval_strategy="ga",
            )

            y_lo[i] = response_y_itvl.lo
            y_hi[i] = response_y_itvl.hi

        return prob_proxy_input, y_lo, y_hi
        # with (ɑ, y_lo) and (ɑ, y_hi), train 2 level-2 surrogate models

    def l2_get_response(
        self, meta_model_lo=None, meta_model_hi=None, ipm_model=None, n=1000_000
    ):
        """with l2 model, predict the vector of probability to get the quantile"""

        if not (
            (meta_model_lo is not None and meta_model_hi is not None)
            or (ipm_model is not None)
        ):
            raise ValueError(
                "You must provide either both meta_model_lo and meta_model_hi, or ipm_model."
            )

        # Xnew
        alpha_new_ = self.dependency.u_sample(n=n, random_state=self.random_state)

        if ipm_model is not None:
            mean_hi, mean_lo = ipm_model(
                alpha_new_
            )  # not technically mean_lo but reuse the name for convenience
        else:
            mean_lo = meta_model_lo(X_new=alpha_new_)
            mean_hi = meta_model_hi(X_new=alpha_new_)

        q_l, _ = pba.get_ecdf(np.sort(mean_lo))
        q_r, _ = pba.get_ecdf(np.sort(mean_hi))
        response = Staircase(left=q_l, right=q_r)
        return response
