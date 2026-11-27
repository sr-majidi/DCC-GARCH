import numpy as np
import pandas as pd

from pathlib import Path

def ret_filter(r, max_jump, max_stale):
    pass

ROOT = Path(__file__).resolve().parent
SP500_FILE = ROOT / 'data' / 'sp500.csv'
NASDAQ_FILE = ROOT / 'data' / 'nasdaq_composite.csv'
OUT = ROOT / 'data' / 'universe'
FIRST_DATE = '2009-12-18'      # first estimation date (a trading day)
WINDOW, STEP, HOLD, N = 1250, 63, 63, 1000
MAX_JUMP = 1.0                 # exclude stocks with a daily |log(1 + r)| above this ...
MAX_ZERO_RUN = 2               # ... or with more than this many consecutive zero returns
COLS = ['PERMNO', 'PERMCO', 'USIncFlg', 'DlyCalDt', 'DlyPrc', 'DlyRet', 'DlyVol', 'MbrStartDt', 'MbrEndDt']

def load(path, member_value):
    parts = []
    for ch in pd.read_csv(path, usecols=COLS, parse_dates=['DlyCalDt', 'MbrStartDt', 'MbrEndDt'], na_values=['NA', ''],
                          dtype={'USIncFlg': str}, chunksize=3_000_000):
        member = (ch.DlyCalDt >= ch.MbrStartDt) & (ch.DlyCalDt <= ch.MbrEndDt)
        parts.append(pd.DataFrame({'date': ch.DlyCalDt, 'id': ch.PERMNO, 'co': ch.PERMCO, 'ret': ch.DlyRet,
                                   'dv': ch.DlyPrc.abs() * ch.DlyVol, 'nonus': (ch.USIncFlg == 'N').astype('float32'),
                                   'key': np.where(member, member_value, 0).astype('float32')}))
    return pd.concat(parts)

def main():
    d = pd.concat([load(SP500_FILE, 2), load(NASDAQ_FILE, 1)])
    company = d.groupby('id').co.last()                                   # PERMNO -> PERMCO
    d = d.sort_values('key', ascending=False, kind='stable').drop_duplicates(['date', 'id'])   # S&P row wins
    ret, dv, key, nonus = (d.pivot(index='date', columns='id', values=v) for v in ['ret', 'dv', 'key', 'nonus'])
    del d
    dates = ret.index
    OUT.mkdir(parents=True, exist_ok=True)

    for tau in range(int(dates.searchsorted(pd.Timestamp(FIRST_DATE))), len(dates) - HOLD, STEP):
        t, win = dates[tau], slice(tau - WINDOW + 1, tau + 1)
        R = ret.iloc[win]
        with np.errstate(all='ignore'):
            ok = ((key.iloc[tau] > 0) & (nonus.iloc[tau] != 1)                      # member on t, US incorporated
                  & R.notna().all() & (R > -1).all()                                # complete history, no -100% return
                  & (np.log1p(R).abs().max() <= MAX_JUMP)                           # no jumps
                  & ((R == 0).rolling(MAX_ZERO_RUN + 1).sum().max() <= MAX_ZERO_RUN))   # no stale prices
        ids = ok[ok].index
        info = pd.DataFrame({'co': company[ids], 'dv': dv.iloc[win][ids].mean(), 'sp500': key.iloc[tau][ids] == 2})
        info = info.sort_values(['sp500', 'dv'], ascending=False).drop_duplicates('co')   # one PERMNO per company
        sp, nq = info[info.sp500], info[~info.sp500]
        u = pd.concat([sp, nq.head(max(N - len(sp), 0))]).sort_values('dv', ascending=False)   # S&P first, fill with Nasdaq
        assert len(u) == N, f'{t.date()}: only {len(u)} eligible stocks (need {N})'
        u.insert(0, 'rank', range(1, N + 1))
        u.index.name = 'PERMNO'
        tag = f'{t:%Y-%m-%d}'
        R[u.index].to_csv(OUT / f'est_{tag}.csv')
        ret.iloc[tau + 1:tau + 1 + HOLD][u.index].to_csv(OUT / f'hold_{tag}.csv')
        u.to_csv(OUT / f'members_{tag}.csv')
        print(f'{tag}: {len(sp)} S&P 500 members + {N - len(sp)} Nasdaq stocks')


if __name__ == '__main__':
    main()