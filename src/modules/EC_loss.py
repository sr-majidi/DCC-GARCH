'''
    This script implements the loss function defined in Engle, Ledoit and Wolf (2019) sec. 4
    as the difference between the out-of-sample portfolio variances of the portfolio
    obtained from estimator H_t and the one obtained from the true covariance matrix H.
    
'''

import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg.blas import dtrmm
from scipy.linalg.lapack import dpotrf, dpotri, dtrtri
from scipy.optimize import minimize
from scipy.signal import lfilter

def _chol_lower(A):
    """Lower Cholesky factor of a symmetric positive definite matrix (LAPACK potrf)."""
    L, info = dpotrf(A, lower=1, clean=1)
    if info != 0:
        raise np.linalg.LinAlgError('Matrix is not positive definite.')
    return L


def true_cov_factors(H):
    """Estimator-independent part of the loss for a true covariance matrix H:
    its lower Cholesky factor L_H and Tr(H^-1) = ||L_H^-1||_F^2.

    Compute once per time point and pass to loss(..., true_factors=...) to evaluate
    many estimators against the same H without refactoring it."""
    L = _chol_lower(H)
    L_inv, info = dtrtri(L, lower=1)
    if info != 0:
        raise np.linalg.LinAlgError('Singular Cholesky factor.')
    return L, np.sum(np.tril(L_inv) ** 2)

def EC_loss(H_hat, H, true_factors=None):
    """L(H_hat, H) = [Tr(H_hat^-1 H H_hat^-1)/N] / [Tr(H_hat^-1)/N]^2 - 1/[Tr(H^-1)/N].

    Non-negative, and zero iff H_hat is proportional to H. Pass
    true_factors=true_cov_factors(H) to reuse the factorization of H across estimators.

    With H = L L' and A = H_hat^-1 (symmetric), Tr(A H A) = ||L' A||_F^2, a triangular
    product, and A comes from a Cholesky factorization plus potri (about N^3 flops in total).
    """
    N = H.shape[0]
    L, tr_H_inv = true_cov_factors(H) if true_factors is None else true_factors
    H_hat_inv, info = dpotri(_chol_lower(H_hat), lower=1)
    if info != 0:
        raise np.linalg.LinAlgError('Could not invert H_hat.')
    A = np.tril(H_hat_inv)
    A = A + np.tril(A, -1).T  # potri returns the lower triangle only
    num = np.sum(dtrmm(1.0, L, A, side=0, lower=1, trans_a=1) ** 2) / N
    den = (np.trace(A) / N) ** 2
    return num / den - N / tr_H_inv