'''
    This script implements the simulation scenario in Pakel et al. (2014) "The working paper version"
'''

import numpy as np
from scipy.stats import truncnorm


def sim_single_corr(ndim, rng=None, mean=0.5, sd=0.1, trunc=4.0):
    """Single-factor correlation matrix S_ij = pi_i pi_j (i != j), S_ii = 1,
    with loadings pi_i ~ N(mean, sd^2) truncated at +/- trunc standard deviations."""
    rng = np.random.default_rng(rng)
    pi = truncnorm.rvs(-trunc, trunc, loc=mean, scale=sd, size=ndim, random_state=rng)
    S = np.outer(pi, pi)
    np.fill_diagonal(S, 1.0)
    return S