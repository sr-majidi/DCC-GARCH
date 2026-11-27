"""
Monte Carlo study of the DCC estimators

Data-generating process: univariate GARCH(1,1) with (omega, a, b) = (0.01, 0.05, 0.9) for every asset, scalar DCC with
(alpha, beta) = (0.05, 0.93), Gaussian innovations, single-factor correlation matrix S (drawn once per N and held fixed
across replications). T + BURNIN observations are simulated and the first BURNIN are discarded.
Estimators: {full, composite} likelihood x {sample, linear shrinkage, non-linear shrinkage} correlation target.
Loss: average over the last LOSS_OBS observations of L(H_hat_t, H_t), with H_hat_t the in-sample conditional covariance
implied by the fitted model and H_t the true one (all estimators are scored in one pass, so H_t is factorized once).

Output: OUT/results.csv  one row per replication x estimator (alpha, beta, loss, converged, fit_sec), appended after every
                         replication, so an interrupted run resumes where it stopped
        OUT/summary.csv  mean, bias and standard deviation of alpha and beta and the mean and sd of the loss, by N and estimator
"""
import sys
import time
import warnings
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import os

import numpy as np
import pandas as pd
import hashlib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.models.DCC import DCC                                                 # noqa: E402
from src.modules.EC_loss import EC_loss, true_cov_factors
from src.modules.corr_simulator import sim_single_corr
from src.modules.corr_target import make_target
from src.utils.pl_proc import init_worker

# ------------------------------------------------ settings
OUT = ROOT / 'data' / 'monte_carlo'
SIZES = (100, 500, 1000)
TOTAL_REPS = 100_000                    # number of replications for each N is TOTAL_REPS / N
T, BURNIN, LOSS_OBS = 1250, 500, 252
DCC_PARAMS = np.array([0.05, 0.93])
GARCH_PARAMS = np.array([0.01, 0.05, 0.9])
LIKELIHOODS = ('mle', 'mscle')
TARGETS = ('sample', 'linear', 'nls')   # 'nls' needs MATLAB (QuEST); remove it from this list to skip it
SEED = 1776   #
LOOSE_TOL_FROM_N = 800                  # panels with N >= 800 use the looser SLSQP tolerance 1e-4
QUEST_DIR = ROOT / 'src' / 'modules' / 'QuEST_v027'
WORKERS = 4                             # parallel worker processes, one replication each (a worker with 'nls' starts a MATLAB engine)
THREADS_PER_WORKER = max(1, (os.cpu_count() or 1) // WORKERS)   # BLAS threads inside each worker

@lru_cache(maxsize=None)
def correlation_matrix(size):
    """The single-factor S of this N: one draw per N, fixed across replications (and identical in every worker)."""
    return sim_single_corr(size, np.random.default_rng([SEED, size]))

def replication(size, rep):
    """Simulate one data set, fit the six estimators and score them. Returns one dict per estimator."""
    sim = DCC()
    X = sim.simulate(T, size, BURNIN, correlation_matrix(size), DCC_PARAMS, GARCH_PARAMS, rng=np.random.default_rng([SEED, size, rep]))
    cache, fits = {}, {}
    for target in TARGETS:
        for likelihood in LIKELIHOODS:
            t0 = time.perf_counter()
            model = DCC(likelihood=likelihood, target=make_target(target, cache))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X, tol=1e-4 if size >= LOOSE_TOL_FROM_N else None)
            fits[(likelihood, target)] = (model, time.perf_counter() - t0)

    start = T - LOSS_OBS
    estimated = {k: m.iter_cond_cov(start) for k, (m, _) in fits.items()}   # H_hat_t, t >= start
    total = dict.fromkeys(fits, 0.0)
    for _, H in sim.iter_true_cov(start):                                    # true H_t, t >= start
        factors = true_cov_factors(H)
        for k, it in estimated.items():
            H_hat = next(it)[1]
            try:
                total[k] += EC_loss(H_hat, H, factors)
            except np.linalg.LinAlgError:                                    # H_hat not positive definite
                total[k] = np.nan
    return [dict(size=size, rep=rep, likelihood=k[0], target=k[1], alpha=m.params_[0], beta=m.params_[1],
                 loss=total[k] / LOSS_OBS, converged=bool(m.opt_.success), fit_sec=sec)
            for k, (m, sec) in fits.items()]


def summarize():
    res = pd.read_csv(OUT / 'results.csv')
    g = res.groupby(['size', 'likelihood', 'target'])
    out = pd.DataFrame({'reps': g.size(),
                        'alpha_mean': g.alpha.mean(), 'alpha_bias': g.alpha.mean() - DCC_PARAMS[0], 'alpha_sd': g.alpha.std(),
                        'beta_mean': g.beta.mean(), 'beta_bias': g.beta.mean() - DCC_PARAMS[1], 'beta_sd': g.beta.std(),
                        'loss_mean': g.loss.mean(), 'loss_sd': g.loss.std()})
    out.to_csv(OUT / 'summary.csv')
    print(out.round(4).to_string())


def monte_carlo():
    OUT.mkdir(parents=True, exist_ok=True)
    results = OUT / 'results.csv'
    done = set(map(tuple, pd.read_csv(results)[['size', 'rep']].to_numpy())) if results.exists() else set()
    tasks = [(size, rep) for size in sorted(SIZES, reverse=True) for rep in range(TOTAL_REPS // size)
             if (size, rep) not in done]                                 # slowest (largest N) first
    print(f'{len(tasks)} replications, {WORKERS} workers x {THREADS_PER_WORKER} threads', flush=True)
    with ProcessPoolExecutor(WORKERS, initializer=init_worker, initargs=(THREADS_PER_WORKER,)) as pool:
        futures = {pool.submit(replication, size, rep): (size, rep) for size, rep in tasks}
        for future in as_completed(futures):
            size, rep = futures[future]
            try:
                rows = future.result()
            except Exception as err:                                     # keep going; the replication is retried next run
                print(f'N={size} replication {rep + 1}: FAILED ({err})', flush=True)
                continue
            pd.DataFrame(rows).to_csv(results, mode='a', header=not results.exists(), index=False)   # only the parent writes
            print(f'N={size} replication {rep + 1}/{TOTAL_REPS // size} done', flush=True)
    summarize()


if __name__ == '__main__':
    monte_carlo()
