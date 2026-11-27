import numpy as np
import pandas as pd
import time
import hashlib
import warnings

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

from src.models.DCC import DCC
from src.modules.corr_target import make_target
from src.utils.pl_proc import init_worker

# ====================================================================================================================
# Estimation of the DCC models on the universe files
#
# For every est_<date>.csv written by build_universe.py, every panel size, correlation target and likelihood it
#   * fits the DCC model on the (demeaned) returns of the first `size` columns (= the `size` most liquid stocks),
#   * appends alpha and beta to  OUT/params.csv,
#   * saves the one-step-ahead forecast of the conditional covariance matrix H_{t+1} (the first day of the holding
#     period) to  OUT/forecasts/H_<date>_N<size>_<likelihood>_<target>.csv  (rows and columns = PERMNO).
# Fits that already have a row in the params estimate file are skipped, so the run can be stopped and restarted at any time.
# ====================================================================================================================
ROOT = Path(__file__).resolve().parent
UNIVERSE = ROOT / 'data' / 'universe'
OUT = ROOT / 'data' / 'estimates'
SIZES = (100, 500, 1000)
LIKELIHOODS = ('mle', 'mscle')
TARGETS = ('sample', 'linear', 'nls')      # 'nls' needs MATLAB (QuEST); remove it from this list to skip it
QUEST_DIR = ROOT / 'src' / 'modules' / 'QuEST_v027'
LOOSE_TOL_FROM_N = 800                     # panels with N >= 800 use the looser SLSQP tolerance 1e-4
WORKERS = 4                                # parallel worker processes (each 'nls' worker starts its own MATLAB engine)
THREADS_PER_WORKER = max(1, (os.cpu_count() or 1) // WORKERS)   # BLAS threads inside each worker

def fit_task(task):
    """One data set (date x size x target): fit the requested likelihoods, save the forecasts, return the params rows."""
    est_file, size, target, likelihoods = task
    date = est_file.stem[len('est_'):]
    returns = pd.read_csv(est_file, index_col=0, usecols=range(size + 1))     # the first `size` stocks
    X = returns.to_numpy(float)
    X = X - X.mean(axis=0)
    ids, cache, rows = returns.columns, {}, []
    for likelihood in likelihoods:
        try:
            t0 = time.perf_counter()
            model = DCC(likelihood=likelihood, target=make_target(target, cache))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X, tol=1e-4 if size >= LOOSE_TOL_FROM_N else None)
            # one-step-ahead forecast: filter through one extra (dummy) day and take the last matrix
            T = X.shape[0]
            H = next(h for _, h in model._iter_cov(np.vstack([X, np.zeros((1, size))]), start=T))
            H = (H + H.T) / 2
            pd.DataFrame(H, index=ids, columns=ids).to_csv(OUT / 'forecasts' / f'H_{date}_N{size}_{likelihood}_{target}.csv',
                                                           float_format='%.8g')
            rows.append(dict(date=date, size=size, likelihood=likelihood, target=target, alpha=model.params_[0],
                             beta=model.params_[1], converged=bool(model.opt_.success), fit_sec=time.perf_counter() - t0))
            print(f'{date} N={size} {likelihood}-{target}: alpha={model.params_[0]:.4f} beta={model.params_[1]:.4f} '
                  f'({time.perf_counter() - t0:.0f}s)', flush=True)
        except Exception as err:                # keep going; a failed fit is not in params.csv and is retried next run
            print(f'{date} N={size} {likelihood}-{target}: FAILED ({err})', flush=True)
    return rows

#def estimate():
#    (OUT / 'forecasts').mkdir(parents=True, exist_ok=True)
#    params_file = OUT / 'params.csv'
#    for est_file in sorted(UNIVERSE.glob('est_*.csv')):
#        date = est_file.stem[len('est_'):]
#        returns = pd.read_csv(est_file, index_col=0)
#        for size in SIZES:
#            X = returns.iloc[:, :size].to_numpy(float)
#            X = X - X.mean(axis=0)
#            ids = returns.columns[:size]
#            cache = {}
#            for target in TARGETS:
#                for likelihood in LIKELIHOODS:
#                    h_file = OUT / 'forecasts' / f'H_{date}_N{size}_{likelihood}_{target}.csv'
#                    if h_file.exists():
#                        continue
#                    try:
#                        t0 = time.perf_counter()
#                        model = DCC(likelihood=likelihood, target=make_target(target, cache))
#                        with warnings.catch_warnings():
#                            warnings.simplefilter('ignore')
#                            model.fit(X, tol=1e-4 if size >= LOOSE_TOL_FROM_N else None)
#                        # one-step-ahead forecast: filter through one extra (dummy) day and take the last matrix
#                        T = X.shape[0]
#                        H = next(h for _, h in model._iter_cov(np.vstack([X, np.zeros((1, size))]), start=T))
#                        H = (H + H.T) / 2
#                        pd.DataFrame(H, index=ids, columns=ids).to_csv(h_file, float_format='%.8g')
#                        row = pd.DataFrame([dict(date=date, size=size, likelihood=likelihood, target=target,
#                                                 alpha=model.params_[0], beta=model.params_[1],
#                                                 converged=bool(model.opt_.success), fit_sec=time.perf_counter() - t0)])
#                        row.to_csv(params_file, mode='a', header=not params_file.exists(), index=False)
#                        print(f'{date} N={size} {likelihood}-{target}: alpha={model.params_[0]:.4f} '
#                              f'beta={model.params_[1]:.4f} ({time.perf_counter() - t0:.0f}s)', flush=True)
#                    except Exception as err:        # keep going; a failed fit has no forecast file and is retried next run
#                        print(f'{date} N={size} {likelihood}-{target}: FAILED ({err})', flush=True)

def estimate():
    (OUT / 'forecasts').mkdir(parents=True, exist_ok=True)
    params_file = OUT / 'params.csv'
    done = set(map(tuple, pd.read_csv(params_file)[['date', 'size', 'likelihood', 'target']].to_numpy())) \
        if params_file.exists() else set()
    tasks = []
    for est_file in sorted(UNIVERSE.glob('est_*.csv')):
        date = est_file.stem[len('est_'):]
        for size in SIZES:
            for target in TARGETS:
                todo = tuple(l for l in LIKELIHOODS if (date, size, l, target) not in done)
                if todo:
                    tasks.append((est_file, size, target, todo))
    tasks.sort(key=lambda task: -task[1])       # slowest (largest N) first, so that the workers finish together
    print(f'{len(tasks)} tasks, {WORKERS} workers x {THREADS_PER_WORKER} threads', flush=True)
    with ProcessPoolExecutor(WORKERS, initializer=init_worker, initargs=(THREADS_PER_WORKER,)) as pool:
        for future in as_completed([pool.submit(fit_task, task) for task in tasks]):
            try:
                rows = future.result()
            except Exception as err:            # e.g. the worker process died
                print(f'task failed: {err}', flush=True)
                continue
            if rows:                            # only the parent writes params.csv
                pd.DataFrame(rows).to_csv(params_file, mode='a', header=not params_file.exists(), index=False)


if __name__ == '__main__':
    estimate()
