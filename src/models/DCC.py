import hashlib
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.signal import lfilter
from statsmodels.tools.validation import array_like, int_like, string_like


#### Univariate GARCH(1,1) log-likelihood function
## The filter returns recursions of the GARCH(1,1) process given the parameters of the process
def _garch_filter(r2, params, v0):
    """Conditional variances of a GARCH(1,1), d2_0 = v0, via a linear filter."""
    omega, a, b = params
    u = np.empty_like(r2)
    u[0] = v0
    u[1:] = omega + a * r2[:-1]
    return lfilter([1.0], [1.0, -b], u)  # d2_t = u_t + b d2_{t-1}

## The negative quasi-log-likelihood function of the GARCH(1,1) process
def _garch_nllf(params, r2, v0):
    omega, a, b = params
    if omega <= 0 or a < 0 or b < 0 or a + b >= 1:
        return 1e10
    d2 = _garch_filter(r2, params, v0)
    return 0.5 * np.mean(np.log(d2) + r2 / d2)

_MARGIN = 1e-6  # numerical margin for the strict inequalities a, b > 0 and a + b < 1

class DCC:
    """
    Parameters
    ----------
    likelihood : {'mle', 'mcle', 'mscle'}
        Full Gaussian likelihood, composite likelihood over all pairs, or over
        contiguous pairs (Pakel et al. 2021).
    target : {'sample', 'linear'} or callable
        Estimator of the correlation-targeting matrix S from the standardized
        residuals eps (T x N). 'sample' = eps'eps/T; 'linear' = Ledoit-Wolf (2004)
        shrinkage toward a scaled identity. A callable f(eps) -> (N x N) array
        plugs in any other estimator, e.g. non-linear shrinkage.
    dist : {'normal'}
    """

    def __init__(self, likelihood='mle', target='sample', dist='normal'):
        self.likelihood = string_like(likelihood, 'likelihood', options=('mle', 'mcle', 'mscle'))
        self.target = target if callable(target) else string_like(target, 'target',
                                                                  options=('sample', 'linear'))
        self._dist = string_like(dist, 'dist')
        if self._dist != 'normal':
            raise NotImplementedError('Currently only the Gaussian likelihood function is implemented.')

        self.endog_simulated = None
        self.sim_ = None
        self.params_ = None          # (alpha, beta)
        self.garch_params_ = None    # (N, 3): omega, a, b per asset
        self.params_history_ = None  # filled by rolling_forecast

    def simulate(self, nobs, ndim, burnin, corr_matrix, dcc_params, garch_params, rng=None):
        """
        Parameters
        ----------
        nobs : int
            Number of observations to simulate.
        ndim : int
            Number of return series to simulate.
        burnin : int
            Burn-in period.
        corr_matrix : np.ndarray
            The intercept (target) correlation matrix of the model.
        dcc_params : np.ndarray
            1-D array of the parameters of the scalar DCC model.
        garch_params : np.ndarray
            1-D array of the parameters of the univariate GARCH(1,1) model.
        rng : numpy.random._generator.Generator, optional
            Numpy Random Generator object to use for the simulations. The default is None.

        Returns
        -------
        np.ndarray
            (nobs x ndim) matrix containing the simulated returns.

        """

        """Simulate nobs observations of a DCC-GARCH(1,1) after a burn-in of `burnin`.
        The initial values in the recursions are as follows:
            Q_0 = S, the unconditional correlation matrix
            d2_0 = omega / (1 - a -b), the unconditional variance of each individual series

        Return value is an (nobs x ndim) matrix of simulated returns."""

        nobs, ndim, burnin = int_like(nobs, 'nobs'), int_like(ndim, 'ndim'), int_like(burnin, 'burnin')
        S = array_like(corr_matrix, 'corr_matrix', ndim=2, shape=(ndim, ndim))
        ## Check if the input target correlation matrix is pd
        np.linalg.cholesky(S)  # raises LinAlgError if S is not positive definite

        alpha, beta = array_like(dcc_params, 'dcc_params', ndim=1, shape=(2,))
        omega, a, b = array_like(garch_params, 'garch_params', ndim=1, shape=(3,))
        rng = np.random.default_rng(rng)

        T = nobs + burnin
        c = 1 - alpha - beta

        ## Empty arrays for the simulation output
        eps = np.empty((T, ndim))
        d2 = np.empty((T, ndim))
        X = np.empty((T, ndim))

        Q = S.copy()
        d2[0] = omega / (1 - a - b)
        for t in range(T):
            if t > 0:
                Q = c * S + alpha * np.outer(eps[t-1], eps[t-1]) + beta * Q
                d2[t] = omega + a * X[t-1]**2 + b * d2[t-1]
            if t == burnin:
                Q_start = Q.copy()  # true Q at the first retained observation
            q = 1 / np.sqrt(np.diag(Q))
            #R = Q * np.outer(q, q)
            eps[t] = np.linalg.cholesky(Q * np.outer(q, q)) @ rng.standard_normal(ndim)
            X[t] = np.sqrt(d2[t]) * eps[t]

        self.sim_ = {'eps': eps[burnin:], 'd2': d2[burnin:], 'Q_start': Q_start, 'S': S,
                     'dcc_params': np.array([alpha, beta]), 'garch_params': np.array([omega, a, b])}
        self.endog_simulated = X[burnin:]
        return self.endog_simulated

    def iter_true_cov(self, start=0):
        """Yield (t, H_t) for the true conditional covariances of the last simulation."""
        if self.sim_ is None:
            raise ValueError('No simulation available; call simulate() first.')
        eps, d2, S = self.sim_['eps'], self.sim_['d2'], self.sim_['S']
        alpha, beta = self.sim_['dcc_params']
        c = 1 - alpha - beta
        Q = self.sim_['Q_start'].copy()
        for t in range(eps.shape[0]):
            if t >= start:
                q = np.sqrt(d2[t] / np.diag(Q))
                yield t, Q * np.outer(q, q)  # D_t R_t D_t
            Q = c * S + alpha * np.outer(eps[t], eps[t]) + beta * Q


    ######## Model Estimation
    def fit(self, endog=None, starting_values=None, method='SLSQP', tol=None, options=None):
        """
        Parameters
        ----------
        endog : np.ndarray, optional
            T x N array of returns. If not passed the output of the last simulation will be used.
        starting_values : np.ndarray, optional
            Starting values of the parameters of the DCC model. The default is None.
        method : string, optional
            DESCRIPTION. The default is 'SLSQP'.
        tol : float, optional
            DESCRIPTION. The default is None.
        options : TYPE, optional
            DESCRIPTION. The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """

        if endog is None:
            if self.endog_simulated is None:
                raise ValueError('Provide endog or call simulate() first.')
            X = self.endog_simulated
        else:
            X = array_like(endog, 'endog', ndim=2)
        T, N = X.shape

        # Step 1: univariate GARCH(1,1) by Gaussian QML, asset by asset
        theta = np.empty((N, 3))
        d2 = np.empty((T, N))
        garch_cons = {'type': 'ineq', 'fun': lambda p: 1 - _MARGIN - p[1] - p[2]}
        for i in range(N):
            r2 = X[:, i]**2
            v0 = r2.mean()
            opt_i = minimize(_garch_nllf, np.array([0.05 * v0, 0.05, 0.9]), args=(r2, v0),
                             method='SLSQP', bounds=[(1e-12, None), (0, 1), (0, 1)],
                             constraints=garch_cons)
            theta[i] = opt_i.x
            d2[:, i] = _garch_filter(r2, opt_i.x, v0)
        eps = X / np.sqrt(d2)

        # Step 2: correlation targeting
        S = self._target_matrix(eps)

        # Step 3: DCC parameters
        if self.likelihood == 'mle':
            func, args = self._nllf_full, (eps, S)
        else:
            if self.likelihood == 'mcle':
                i, j = np.triu_indices(N, 1)
            else:
                i = np.arange(N - 1)
                j = i + 1
            func, args = self._nllf_composite, (eps, S, i, j)

        if starting_values is None:
            sv = np.array([0.05, 0.9])
        else:
            sv = array_like(starting_values, 'starting_values', ndim=1, shape=(2,))
            if np.any(sv <= 0) or sv.sum() >= 1:
                raise ValueError('Starting values must satisfy alpha, beta > 0 and alpha + beta < 1.')

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Values in x were outside bounds during a minimize step',
                                    RuntimeWarning)
            opt = minimize(func, sv, args=args, method=method,
                           bounds=[(_MARGIN, 1 - _MARGIN)] * 2,
                           constraints={'type': 'ineq', 'fun': lambda p: 1 - _MARGIN - p.sum()},
                           tol=tol, options={} if options is None else options)
        if not opt.success:
            warnings.warn(f'DCC optimization did not converge: {opt.message}', RuntimeWarning)

        self.endog_, self.eps_, self.d2_, self.S_ = X, eps, d2, S
        self.garch_params_, self.params_, self.opt_ = theta, opt.x, opt
        return self

    def _target_matrix(self, eps):
        ### TODO: scale the diagonal entries of the correlation matrices
        if callable(self.target):
            return np.asarray(self.target(eps), dtype=float)
        T, N = eps.shape
        S = eps.T @ eps / T
        if self.target == 'sample':
            return S
        # Ledoit-Wolf (2004) linear shrinkage toward mu * I (norm ||A||^2 = Tr(AA')/N)
        mu = np.trace(S) / N
        delta2 = np.sum((S - mu * np.eye(N))**2) / N
        # sum_t ||x_t x_t' - S||^2 = sum_t (x_t'x_t)^2 - 2 sum_t x_t'S x_t + T ||S||^2
        beta2_bar = (np.sum(np.sum(eps**2, axis=1)**2) - 2 * np.sum((eps @ S) * eps)
                     + T * np.sum(S**2)) / (N * T**2)
        s = min(beta2_bar, delta2) / delta2
        return s * mu * np.eye(N) + (1 - s) * S

    @staticmethod
    def _nllf_full(params, eps, S):
        """(1/2T) sum_t [log|R_t| + eps_t' R_t^-1 eps_t - eps_t' eps_t]; R_t built on the fly."""
        alpha, beta = params
        if alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        c = 1 - alpha - beta
        T = eps.shape[0]
        Q = S.copy()
        nllf = 0.0
        for t in range(T):
            q = 1 / np.sqrt(np.diag(Q))
            try:
                L = np.linalg.cholesky(Q * np.outer(q, q))
            except np.linalg.LinAlgError:
                return 1e10
            z = solve_triangular(L, eps[t], lower=True)
            nllf += 2 * np.sum(np.log(np.diag(L))) + z @ z - eps[t] @ eps[t]
            Q = c * S + alpha * np.outer(eps[t], eps[t]) + beta * Q
        return nllf / (2 * T)

    @staticmethod
    def _nllf_composite(params, eps, S, i, j):
        """Composite likelihood over the pairs (i[k], j[k]). Only the entries of Q
        needed for the pairs are propagated, so the cost is O(T K), not O(T N^2)."""
        alpha, beta = params
        if alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        c = 1 - alpha - beta
        T = eps.shape[0]
        q_diag, q_off = np.diag(S).copy(), S[i, j].copy()
        s_diag, s_off = c * np.diag(S), c * S[i, j]
        nllf = 0.0
        for t in range(T):
            rho = q_off / np.sqrt(q_diag[i] * q_diag[j])
            r_det = 1 - rho**2
            if np.any(r_det <= 0):
                return 1e10
            e_i, e_j = eps[t, i], eps[t, j]
            nllf += np.sum(np.log(r_det) + (e_i**2 - 2 * rho * e_i * e_j + e_j**2) / r_det
                           - (e_i**2 + e_j**2))
            q_diag = s_diag + alpha * eps[t]**2 + beta * q_diag
            q_off = s_off + alpha * e_i * e_j + beta * q_off
        return nllf / (2 * T * len(i))


    ##### Forecasts of the Conditional Covariance MAtrix H_t
    def _iter_cov(self, X, start=0):
        """Run the fitted GARCH and DCC filters over X (same initialization as in fit)
        and yield (t, H_hat_t) for t >= start. H_hat_t uses information up to t-1."""
        omega, a, b = self.garch_params_.T
        alpha, beta = self.params_
        c = 1 - alpha - beta
        S = self.S_
        d2 = self.d2_[0].copy()
        Q = S.copy()
        for t in range(X.shape[0]):
            if t > 0:
                d2 = omega + a * X[t-1]**2 + b * d2
                Q = c * S + alpha * np.outer(e, e) + beta * Q
            e = X[t] / np.sqrt(d2)
            if t >= start:
                q = np.sqrt(d2 / np.diag(Q))
                yield t, Q * np.outer(q, q)

    def iter_cond_cov(self, start=0):
        """Yield (t, H_hat_t) over the estimation sample for t >= start."""
        if self.params_ is None:
            raise ValueError('Call fit() first.')
        return self._iter_cov(self.endog_, start)

    def rolling_forecast(self, endog, window, update_freq, **fit_kwargs):
        """Re-estimate on a rolling window of fixed length `window` every `update_freq`
        observations, and yield (t, H_hat_t) one-step-ahead forecasts for t >= window.

        This is a generator: the fits run as it is iterated. Estimates are stored
        in self.params_history_."""
        X = array_like(endog, 'endog', ndim=2)
        window, update_freq = int_like(window, 'window'), int_like(update_freq, 'update_freq')
        T = X.shape[0]
        if not 0 < window < T or update_freq < 1:
            raise ValueError('Require 0 < window < nobs and update_freq >= 1.')
        self.params_history_ = []
        for tau in range(window, T, update_freq):
            self.fit(X[tau - window:tau], **fit_kwargs)
            self.params_history_.append({'t': tau, 'dcc': self.params_.copy(),
                                         'garch': self.garch_params_.copy()})
            t_end = min(tau + update_freq, T)
            for t, H in self._iter_cov(X[tau - window:t_end], start=window):
                yield tau - window + t, H

    # ------------------------------------------------------------------ #
    def summary(self):
        if self.params_ is None:
            raise ValueError('Call fit() first.')
        target = self.target if isinstance(self.target, str) else getattr(self.target, '__name__', 'callable')
        g = self.garch_params_
        lines = [f'DCC-GARCH(1,1)   likelihood = {self.likelihood}   target = {target}',
                 f'  T = {self.endog_.shape[0]}, N = {self.endog_.shape[1]}',
                 f'  alpha = {self.params_[0]:.4f}   beta = {self.params_[1]:.4f}'
                 f'   (converged: {self.opt_.success})',
                 '  Univariate GARCH(1,1), cross-sectional mean (sd):']
        for k, name in enumerate(('omega', 'a', 'b')):
            lines.append(f'    {name:<5} = {g[:, k].mean():.4f} ({g[:, k].std():.4f})')
        return '\n'.join(lines)