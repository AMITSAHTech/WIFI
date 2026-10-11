import pickle, time, numpy as np, sys
from env import *; from policies import *; from runlib import *
B = tuned()
SEEDS = list(range(0, 30)); DAYS = 3
C = ctors(B); AB = ablations(B)
names = list(C) + [k for k in AB if k != 'AIZen']
res = {n: [] for n in names}
pool = {n: dict(T=[], R=[]) for n in ['Oracle', 'Native', 'KV', 'EpsSticky', 'TSStat', 'AIZen']}
t0 = time.time()
for s in SEEDS:
    env = make_env(s, days=DAYS)
    for n in C:
        m = run_assoc(env, C[n](s), return_trace=(n in pool))
        if n in pool:
            tr = m.pop('_trace')
            pool[n]['T'].append(tr['Teff'][::5].astype(np.float32))
            pool[n]['R'].append(env['rtt_samp'][np.arange(env['T']), tr['arm'], 0][::2].astype(np.float32))
        res[n].append(m)
    for n in AB:
        if n == 'AIZen': continue
        res[n].append(run_assoc(env, AB[n](s)))
    print('seed', s, '%.0fs' % (time.time() - t0), flush=True)
pickle.dump(dict(res=res, pool={k: dict(T=np.concatenate(v['T']), R=np.concatenate(v['R'])) for k, v in pool.items()}, seeds=SEEDS, days=DAYS, tuned=B), open('exp1.pkl', 'wb'))
print('done')
