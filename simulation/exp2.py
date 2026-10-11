"""Experiment 2: channel selection through the optional router actuator (per band)."""
import itertools, json, pickle, time, numpy as np
from env import *; from env2 import *; from policies import *; from runlib import summarize

def csa(p): p.ho_kind = 'csa'; return p
GR = {
 'EpsSticky': (lambda **k: csa(EpsSticky(**k)), dict(eps=[0.02, 0.05, 0.1], alpha=[0.1, 0.3], dwell=[10, 30])),
 'DUCB':      (lambda **k: csa(DUCB(**k)),      dict(gamma=[0.99, 0.995, 0.999], dwell=[10, 30])),
 'SWUCB':     (lambda **k: csa(SWUCB(**k)),     dict(tau=[300, 1000, 3000], dwell=[10, 30])),
 'TSStat':    (lambda **k: csa(TSStat(**k)),    dict(dwell=[10, 30])),
 'SWTS':      (lambda **k: csa(SWTS(**k)),      dict(tau=[300, 1000, 3000], dwell=[10, 30])),
 'AIZen':     (lambda **k: csa(AIZen(**k)),     dict(gamma=[0.99, 0.999], G0=[1.0, 4.0], kappa=[2.0, 8.0], dwell=[10])),
 'ACS':       (lambda **k: ACS(**k),            dict(period_h=[1.0, 6.0, 24.0])),
}
VAL = list(range(200, 206)); TEST = list(range(0, 20)); DAYS = 3
out = {}
t0 = time.time()
for bd in (0, 1):
    venv = [make_chan_env(make_env(s, days=2), bd) for s in VAL]
    best = {}
    for name, (mk, g) in GR.items():
        keys = list(g); rs = []
        for vals in itertools.product(*[g[k] for k in keys]):
            kw = dict(zip(keys, vals))
            if name not in ('ACS',): pass
            r = np.mean([run_assoc(e, mk(seed=i, **kw) if name != 'ACS' else mk(**kw))['reward'] for i, e in enumerate(venv)])
            rs.append((float(r), kw))
        rs.sort(key=lambda x: -x[0]); best[name] = rs[0][1]
        print('band', bd, name, 'best', rs[0], '%.0fs' % (time.time() - t0), flush=True)
    ctor = {'Oracle': lambda s: csa(Oracle()), 'Static (factory default)': lambda s: Static(), 'AutoBoot': lambda s: AutoBoot(),
            'ACS (tuned period)': lambda s: ACS(**best['ACS'])}
    for n in ['EpsSticky', 'DUCB', 'SWUCB', 'TSStat', 'SWTS', 'AIZen']:
        ctor[n] = (lambda n: lambda s: GR[n][0](seed=s, **best[n]))(n)
    res = {n: [] for n in ctor}
    for s in TEST:
        e = make_chan_env(make_env(s, days=DAYS), bd)
        for n in ctor: res[n].append(run_assoc(e, ctor[n](s)))
    out[bd] = dict(res=res, best=best)
    pickle.dump(out, open('exp2.pkl', 'wb'))
    print('band', bd, 'done %.0fs' % (time.time() - t0), flush=True)
pickle.dump(out, open('exp2.pkl', 'wb')); print('done')
