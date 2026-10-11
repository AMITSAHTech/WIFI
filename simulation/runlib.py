import json, numpy as np
from policies import *
def tuned(path='tuned.json'): return json.load(open(path))['best']
def _mbb(p): p.ho_kind = 'mbb'; return p
def ctors(B, over=None):
    over = over or {}
    A = dict(B['AIZen']); A.update(over)
    return {
     'Oracle':   lambda s: Oracle(),
     'Native':   lambda s: Native(),
     'RSSI-best':lambda s: RSSIBest(),
     'KV':       lambda s: KV(),
     'EpsSticky':lambda s: EpsSticky(seed=s, **B['EpsSticky']),
     'DUCB':     lambda s: DUCB(seed=s, **B['DUCB']),
     'SWUCB':    lambda s: SWUCB(seed=s, **B['SWUCB']),
     'EpsSticky+MBB': lambda s: _mbb(EpsSticky(seed=s, **B['EpsSticky'])),
     'TSStat':   lambda s: TSStat(seed=s, **B['TSStat']),
     'SWTS':     lambda s: SWTS(seed=s, **B['SWTS']),
     'AIZen':    lambda s: AIZen(seed=s, **A),
    }
def ablations(B):
    A = dict(B['AIZen'])
    def mk(**k):
        d = dict(A); d.update(k); return lambda s: AIZen(seed=s, **d)
    return {
     'AIZen':            mk(),
     '-NN (heuristic prior)': mk(use_nn=False),
     '-prior (kappa=0)': mk(kappa=0.0),
     '-decay':           mk(discount=False),
     '-gate':            mk(use_gate=False),
     '-anomaly':         mk(use_anom=False),
     '-meta-learning':   mk(use_meta=False),
     '-safety':          mk(use_safety=False),
     '-MBB (break-before-make)': mk(mbb=False),
    }
def summarize(vals):
    v = np.asarray(vals, float); n = len(v)
    from scipy import stats
    ci = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return float(v.mean()), float(ci)
