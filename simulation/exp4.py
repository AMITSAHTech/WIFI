"""Experiment 4: sensitivity (environment parameters, gamma, reward weights, handoff mechanism). No re-tuning."""
import pickle, time, numpy as np
from env import *; from policies import *; from runlib import *
B = tuned(); SEEDS = list(range(0, 10)); DAYS = 2
A = dict(B['AIZen'])
def cz(**k): d = dict(A); d.update(k); return lambda s: AIZen(seed=s, **d)
BASE = {'Oracle': lambda s: Oracle(), 'Native': lambda s: Native(), 'KV': lambda s: KV(), 'AIZen': cz(),
        'EpsSticky': lambda s: EpsSticky(seed=s, **B['EpsSticky'])}
out = {}; t0 = time.time()
def sweep(tag, values, key=None, ctor=None, label=None):
    out[tag] = {}
    for v in values:
        over = {key: v} if key else {}
        cs = ctor(v) if ctor else BASE
        envs = [make_env(s, days=DAYS, **over) for s in SEEDS]
        out[tag][v] = {n: [run_assoc(e, c(s)) for s, e in zip(SEEDS, envs)] for n, c in cs.items()}
        pickle.dump(out, open('exp4.pkl', 'wb'))
        print(tag, v, '%.0fs' % (time.time() - t0), flush=True)
sweep('load_scale', [0.5, 1.0, 2.0, 3.0, 4.0, 6.0], 'load_scale')
sweep('sigma_sh', [2.0, 4.0, 8.0], 'sigma_sh')
sweep('n24', [2.6, 3.0, 3.4], 'n24')
sweep('n_nb', [5, 10, 19], 'n_nb')
sweep('p_mbb', [0.0, 0.5, 0.9, 1.0], 'p_mbb', ctor=lambda v: {'Native': BASE['Native'], 'AIZen': BASE['AIZen']})
sweep('gamma', [0.90, 0.95, 0.98, 0.99, 0.998, 1.0], None,
      ctor=lambda g: {'AIZen': cz(gamma=min(g, 0.999999), discount=(g < 1.0))})
W = {'balanced (0.60/0.25/0.15)': (0.60, 0.25, 0.15), 'throughput only': (1.0, 0.0, 0.0), 'latency-heavy (0.40/0.40/0.20)': (0.40, 0.40, 0.20), 'equal (1/3 each)': (1 / 3, 1 / 3, 1 / 3)}
out['weights'] = {}
envs = [make_env(s, days=DAYS) for s in SEEDS]
for k, w in W.items():
    out['weights'][k] = {'AIZen': [run_assoc(e, cz(w=w)(s)) for s, e in zip(SEEDS, envs)]}
    print('weights', k, '%.0fs' % (time.time() - t0), flush=True)
pickle.dump(out, open('exp4.pkl', 'wb')); print('done')
