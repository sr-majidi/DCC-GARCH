import numpy as np
import hashlib

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
QUEST_DIR = ROOT / 'QuEST_v027'
DCC_DIR = ROOT / '..' / 'models'
sys.path.insert(0, str(DCC_DIR))
from DCC import DCC


def to_corr(sigma):
    """Symmetrize and rescale a covariance matrix to a unit diagonal."""
    sigma = (sigma + sigma.T) / 2
    d = 1 / np.sqrt(np.diag(sigma))
    return sigma * np.outer(d, d)

_matlab_engine = None

def nls_matrix(eps):
    """Non-linear shrinkage of Ledoit and Wolf (QuEST function, MATLAB code in src/modules/QuEST_v027)."""
    global _matlab_engine
    import matlab.engine
    if _matlab_engine is None:
        _matlab_engine = matlab.engine.start_matlab()
        _matlab_engine.addpath(_matlab_engine.genpath(str(QUEST_DIR)), nargout=0)
    return np.asarray(_matlab_engine.QuESTimate(matlab.double(eps), 0.0, nargout=9)[0])

def make_target(name, cache):
    """Correlation target f(eps) for the DCC class. The standardized residuals are the same for the full and the
    composite likelihood, so each target is computed once per data set and kept in `cache`."""
    def target(eps):
        key = (name, hashlib.blake2b(np.ascontiguousarray(eps).tobytes(), digest_size=16).digest())
        if key not in cache:
            sigma = nls_matrix(eps) if name == 'nls' else DCC(target=name)._target_matrix(eps)
            cache[key] = to_corr(sigma)
        return cache[key]
    target.__name__ = name
    return target