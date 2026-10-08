"""Which Pareto solution does the monitoring agent deploy after re-optimisation?

Gate-0 finding: the pre-registered rule (minimum fitness error, ties -> fewest features) deploys
brittle 2-5-feature subsets, while a k-NN on all 561 features transfers at 95-100%. The rule is
the agent's action policy, so it shapes every deployed-accuracy number in the paper.
Compare rules with:  python -m tevc.rescore_deploy
"""
import numpy as np


def default_rule(fit_err, ratio):
    """Pre-registered Gate-0 rule: minimum fitness error, ties -> fewer features."""
    return int(np.lexsort((ratio, fit_err))[0])


def choose_deployed(fit_err, ratio):
    """Return the index of the non-dominated solution to deploy.

    fit_err: (P,) error of each solution on the agent's OWN fitness data (synthesised for the new
             speed) -- the agent never sees real labels from the new environment.
    ratio:   (P,) fraction of the 561 features each solution computes (edge compute cost).
    Many solutions tie at fit_err == 0 because one JNU speed is easy to separate.
    """
    # TODO(user): deployment policy, 5-10 lines.
    raise NotImplementedError
