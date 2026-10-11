import pickle, time, numpy as np
from env import *; from policies import *; from runlib import *
B = tuned(); A = dict(B['AIZen'])
class FixedBest:
    ho_kind = 'none'
    def reset(s, env): s.env = env; s.arm = int(np.argmax(env['Ttrue'].mean(0)))
    def act(s, t, cur): return s.arm
    def observe(s, *a): pass
    def aborted(s, t): pass
out = {}; t0 = time.time()
# (a) extra rows on the 30 test seeds
a = {'FixedBest': [], 'AIZen (gamma=0.95)': []}
for s in range(30):
    e = make_env(s, days=3)
    a['FixedBest'].append(run_assoc(e, FixedBest()))
    a['AIZen (gamma=0.95)'].append(run_assoc(e, AIZen(seed=s, **{**A, 'gamma': 0.95})))
out['extra'] = a; pickle.dump(out, open('exp5.pkl', 'wb')); print('a done %.0fs' % (time.time() - t0), flush=True)
# (b) handoff-cost sensitivity
C = ctors(B); SE = range(10)
sel = ['EpsSticky', 'SWUCB', 'TSStat', 'AIZen']
out['dbbm'] = {}
for d in (0.45, 2.0, 5.0, 10.0):
    envs = [make_env(s, days=2, d_bbm=d) for s in SE]
    out['dbbm'][d] = {n: [run_assoc(e, C[n](s)) for s, e in zip(SE, envs)] for n in sel}
    out['dbbm'][d]['AIZen (no MBB)'] = [run_assoc(e, AIZen(seed=s, **{**A, 'mbb': False})) for s, e in zip(SE, envs)]
    pickle.dump(out, open('exp5.pkl', 'wb')); print('dbbm', d, '%.0fs' % (time.time() - t0), flush=True)
# (c) heavy-tailed traffic
envs = [make_env(s, days=2, traffic='pareto') for s in SE]
out['pareto'] = {n: [run_assoc(e, C[n](s)) for s, e in zip(SE, envs)] for n in ['Oracle', 'Native', 'RSSI-best', 'KV', 'EpsSticky', 'TSStat', 'AIZen']}
pickle.dump(out, open('exp5.pkl', 'wb')); print('done %.0fs' % (time.time() - t0), flush=True)
