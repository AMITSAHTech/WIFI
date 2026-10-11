"""Experiment 3: M independent clients share the home BSSIDs (congestion game)."""
import pickle, time, numpy as np
from env import *; from policies import *; from runlib import tuned

B = tuned()
def pol_ctors():
    A = dict(B['AIZen'])
    return {
      'Native':    lambda s: Native(),
      'Greedy':    lambda s: EpsSticky(eps=0.0, alpha=0.3, dwell=1, seed=s),
      'EpsSticky': lambda s: EpsSticky(seed=s, **B['EpsSticky']),
      'TSStat':    lambda s: TSStat(seed=s, **B['TSStat']),
      'AIZen':     lambda s: AIZen(seed=s, **A),
      'AIZen (no gate)': lambda s: AIZen(seed=s, **{**A, 'use_gate': False, 'dwell': 1}),
    }

def rate_of(P, sinr): return np.minimum(P['rmax'], 0.6 * 20 * 2 * np.log2(1 + 10 ** (sinr / 10)))

def run_multi(env, M, mk, seed, centralized=False):
    P = env['P']; T = env['T']; rng = np.random.default_rng(seed + 31)
    off = rng.normal(0, 3.0, (M, 4)); act = np.zeros((M, T), bool)
    u = rng.random((M, T)); st = rng.random(M) < 0.5
    for t in range(T):
        st = np.where(st, u[:, t] > 0.05, u[:, t] < 0.05); act[:, t] = st
    pols = []; envs = []
    for i in range(M):
        ev = dict(env); ev['rssi'] = env['rssi'] + off[i]
        f = env['feat'].copy(); f[:, :, 10] = np.clip((ev['rssi'] + 90) / 60, 0, 1); ev['feat'] = f
        envs.append(ev)
        if not centralized:
            p = mk(seed * 100 + i); p.reset(ev); pols.append(p)
    cur = np.full(M, -1); Tsum = np.zeros(M); Tcnt = np.zeros(M); hos = np.zeros(M)
    herd = []; Rsum = 0.0
    dis = env['dis']
    for t in range(T):
        a = act[:, t]
        sinr_i = env['sinr'][t][None, :] + off                       # (M,4)
        R = rate_of(P, sinr_i)
        want = cur.copy()
        if centralized:
            n = max(a.sum(), 1)
            if a.any():
                idx = np.where(a)[0]; asg = np.argmax(R[idx], axis=1)
                for _ in range(3):
                    for q, i in enumerate(idx):
                        cnt = np.bincount(asg, minlength=4); cnt[asg[q]] -= 1
                        val = P['eta'] * R[i] * max(0.05, 1) * (1 - env['busy'][t]) / (cnt + 1)
                        asg[q] = int(np.argmax(val))
                for q, i in enumerate(idx): want[i] = asg[q]
            for i in range(M):
                if cur[i] < 0 and want[i] < 0: want[i] = int(np.argmax(R[i]))
        else:
            for i in range(M):
                if cur[i] < 0 or a[i]:
                    w = pols[i].act(t, None if cur[i] < 0 else int(cur[i]))
                    want[i] = w
        D = np.zeros(M)
        for i in range(M):
            if cur[i] < 0: cur[i] = want[i]; continue
            if want[i] != cur[i]:
                k = 'none' if centralized else pols[i].ho_kind
                d = 0.0
                if k == 'ft': d = dis['ft'][t]
                elif k == 'bbm': d = dis['bbm'][t]
                elif k == 'mbb': d = dis['mbb'][t] if dis['u_mbb'][t] < P['p_mbb'] else dis['bbm'][t]
                cur[i] = want[i]; D[i] = min(d, DT)
                if a[i]: hos[i] += 1
        nb = np.bincount(cur[a], minlength=4) if a.any() else np.zeros(4, int)
        if a.sum() >= 2: herd.append(nb.max() / a.sum())
        for i in range(M):
            if not a[i]: continue
            b = cur[i]; n_b = max(nb[b], 1)
            share = (1 - env['busy'][t, b]) / n_b
            Tt = P['eta'] * R[i, b] * max(0.05 / n_b, share)
            Tm = Tt * np.exp(rng.normal(0, 0.15)) * (1 - D[i] / DT)
            rho = np.clip(1 - max(0.05 / n_b, share), 0, 0.985); L = 4 + 10 * rho / (1.02 - rho)
            ploss = 0.001 + 0.05 * rho ** 3 + 0.25 / (1 + np.exp(-(10 - sinr_i[i, b]) / 2.5))
            Pm = 100 * rng.binomial(10, min(ploss, 1)) / 10
            Tsum[i] += Tt * (1 - D[i] / DT); Tcnt[i] += 1
            if not centralized: pols[i].observe(t, int(b), Tm, L, Pm, envs[i]['feat'][t, b])
    ta = Tsum / np.maximum(Tcnt, 1)
    jain = ta.sum() ** 2 / (M * (ta ** 2).sum()) if M > 1 else 1.0
    hours = T * DT / 3600
    return dict(tput=float(ta.mean()), jain=float(jain), ho_per_h=float(hos.mean() / hours),
                herd=float(np.mean(herd)) if herd else np.nan)

if __name__ == '__main__':
    SEEDS = list(range(0, 8)); MS = [2, 4, 8, 12]; res = {}
    t0 = time.time(); ctors = pol_ctors()
    for M in MS:
        res[M] = {n: [] for n in list(ctors) + ['Central oracle']}
        for s in SEEDS:
            env = make_env(s, days=1)
            for n in ctors: res[M][n].append(run_multi(env, M, ctors[n], s))
            res[M]['Central oracle'].append(run_multi(env, M, None, s, centralized=True))
        pickle.dump(res, open('exp3.pkl', 'wb'))
        print('M', M, '%.0fs' % (time.time() - t0), flush=True)
    pickle.dump(res, open('exp3.pkl', 'wb')); print('done')
