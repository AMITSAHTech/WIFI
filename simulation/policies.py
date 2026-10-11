import numpy as np
from collections import deque
from env import DT


def reward(T, L, P, tcap=180.0, w=(0.60, 0.25, 0.15)):
    return float(np.clip(w[0] * T / tcap + w[1] * (1 - L / 200.0) + w[2] * (1 - P / 20.0), 0, 1))


# ----------------------------------------------------------------- baselines
class Native:
    """RSSI-threshold roaming with 5 GHz preference (vendor-typical behaviour)."""
    ho_kind = 'bbm'
    def __init__(s, thr=-75.0, bias5=5.0): s.thr, s.b5 = thr, bias5
    def reset(s, env): s.env = env
    def act(s, t, cur):
        r = s.env['rssi'][t]; sc = r + s.b5 * s.env['band']
        if cur is None: return int(np.argmax(sc))
        return int(np.argmax(sc)) if r[cur] < s.thr else cur
    def observe(s, *a): pass
    def aborted(s, t): pass


class RSSIBest:
    """Always-roam client: strongest RSSI (with 5 GHz bias), 8 dB hysteresis."""
    ho_kind = 'bbm'
    def __init__(s, hyst=8.0, bias5=5.0): s.h, s.b5 = hyst, bias5
    def reset(s, env): s.env = env
    def act(s, t, cur):
        sc = s.env['rssi'][t] + s.b5 * s.env['band']; b = int(np.argmax(sc))
        if cur is None: return b
        return b if sc[b] > sc[cur] + s.h else cur
    def observe(s, *a): pass
    def aborted(s, t): pass


class KV(Native):
    """Native + AP-assisted steering (802.11k neighbour reports, 802.11v BTM, 802.11r fast transition)."""
    ho_kind = 'ft'
    def __init__(s, thr=-75.0, bias5=5.0, btm_rssi=-70.0, load_thr=0.80, p_acc=0.7):
        super().__init__(thr, bias5); s.btm, s.lt, s.pa = btm_rssi, load_thr, p_acc
    def act(s, t, cur):
        e = s.env; r = e['rssi'][t]
        if cur is None: return super().act(t, cur)
        same_ap = np.where(e['arms_ap'] == e['arms_ap'][cur])[0]   # same ESS: every home BSSID
        if r[cur] < s.btm or e['busy'][t, cur] > s.lt:
            ok = np.where(r > s.thr)[0]
            if len(ok):
                cand = ok[np.argmin(e['busy'][t, ok] - 0.01 * r[ok])]
                if cand != cur and e['dis']['u_acc'][t] < s.pa: return int(cand)
        return super().act(t, cur)


class Oracle:
    ho_kind = 'none'
    def reset(s, env): s.env = env
    def act(s, t, cur): return int(np.argmax(s.env['Ttrue'][t]))
    def observe(s, *a): pass
    def aborted(s, t): pass


class _Base:
    ho_kind = 'bbm'
    def __init__(s, dwell=4, seed=0): s.dwell, s.rng, s.seed = dwell, np.random.default_rng(seed + 991), seed
    def reset(s, env): s.env = env; s.K = env['Ttrue'].shape[1]; s.last = -10 ** 9; s.rng = np.random.default_rng(s.seed + 991); s._init()
    def _init(s): pass
    def aborted(s, t): pass
    def _gate(s, t, cur, cand):
        if cur is None: return cand
        if cand != cur and t - s.last >= s.dwell:
            s.last = t; return cand
        return cur


class EpsSticky(_Base):
    """epsilon-greedy with stickiness (adapted from Carrascosa & Bellalta)."""
    def __init__(s, eps=0.1, alpha=0.3, dwell=4, seed=0): super().__init__(dwell, seed); s.eps, s.al = eps, alpha
    def _init(s): s.q = np.ones(s.K) * 0.8
    def act(s, t, cur):
        cand = int(s.rng.integers(0, s.K)) if s.rng.random() < s.eps else int(np.argmax(s.q))
        if cur is None: s.last = t; return cand
        return s._gate(t, cur, cand)
    def observe(s, t, arm, T, L, P, x):
        r = reward(T, L, P, s.env['P']['tcap']); s.q[arm] += s.al * (r - s.q[arm])


class DUCB(_Base):
    def __init__(s, gamma=0.99, dwell=4, xi=0.5, seed=0): super().__init__(dwell, seed); s.g, s.xi = gamma, xi
    def _init(s): s.N = np.zeros(s.K); s.S = np.zeros(s.K)
    def act(s, t, cur):
        n = s.N.sum()
        idx = np.where(s.N > 1e-9, s.S / np.maximum(s.N, 1e-9) + np.sqrt(s.xi * np.log(max(n, 2)) / np.maximum(s.N, 1e-9)), 1e9)
        c = int(np.argmax(idx))
        if cur is None: s.last = t; return c
        return s._gate(t, cur, c)
    def observe(s, t, arm, T, L, P, x):
        s.N *= s.g; s.S *= s.g; s.N[arm] += 1; s.S[arm] += reward(T, L, P, s.env['P']['tcap'])


class SWUCB(_Base):
    def __init__(s, tau=300, dwell=4, xi=0.5, seed=0): super().__init__(dwell, seed); s.tau, s.xi = tau, xi
    def _init(s): s.N = np.zeros(s.K); s.S = np.zeros(s.K); s.h = deque()
    def act(s, t, cur):
        n = min(t + 1, s.tau)
        idx = np.where(s.N > 0, s.S / np.maximum(s.N, 1) + np.sqrt(s.xi * np.log(max(n, 2)) / np.maximum(s.N, 1)), 1e9)
        c = int(np.argmax(idx))
        if cur is None: s.last = t; return c
        return s._gate(t, cur, c)
    def observe(s, t, arm, T, L, P, x):
        r = reward(T, L, P, s.env['P']['tcap']); s.h.append((t, arm, r)); s.N[arm] += 1; s.S[arm] += r
        while s.h and s.h[0][0] <= t - s.tau:
            _, a, rr = s.h.popleft(); s.N[a] -= 1; s.S[a] -= rr


class TSStat(_Base):
    def _init(s): s.a = np.ones(s.K); s.b = np.ones(s.K)
    def act(s, t, cur):
        c = int(np.argmax(s.rng.beta(s.a, s.b)))
        if cur is None: s.last = t; return c
        return s._gate(t, cur, c)
    def observe(s, t, arm, T, L, P, x):
        r = reward(T, L, P, s.env['P']['tcap']); s.a[arm] += r; s.b[arm] += 1 - r


class SWTS(_Base):
    """Sliding-window Beta Thompson sampling (after Trovo et al.)."""
    def __init__(s, tau=300, dwell=4, seed=0): super().__init__(dwell, seed); s.tau = tau
    def _init(s): s.a = np.ones(s.K); s.b = np.ones(s.K); s.h = deque()
    def act(s, t, cur):
        c = int(np.argmax(s.rng.beta(s.a, s.b)))
        if cur is None: s.last = t; return c
        return s._gate(t, cur, c)
    def observe(s, t, arm, T, L, P, x):
        r = reward(T, L, P, s.env['P']['tcap']); s.h.append((t, arm, r)); s.a[arm] += r; s.b[arm] += 1 - r
        while s.h and s.h[0][0] <= t - s.tau:
            _, a, rr = s.h.popleft(); s.a[a] -= rr; s.b[a] -= 1 - rr


# ----------------------------------------------------------------- Wi-Fi AI Zen
class NN:
    def __init__(s, nin, nh, rng, lr=0.01, lam=1e-4):
        a = np.sqrt(6.0 / (nin + nh)); s.W1 = rng.uniform(-a, a, (nh, nin)); s.b1 = np.zeros(nh)
        a2 = np.sqrt(6.0 / (nh + 1)); s.W2 = rng.uniform(-a2, a2, (1, nh)); s.b2 = np.array([0.7])
        s.lr, s.lam = lr, lam; s.t = 0
        s.m = [np.zeros_like(p) for p in (s.W1, s.b1, s.W2, s.b2)]; s.v = [np.zeros_like(p) for p in (s.W1, s.b1, s.W2, s.b2)]
    def predict(s, X):
        H = np.maximum(0, X @ s.W1.T + s.b1); return np.clip((H @ s.W2.T + s.b2)[:, 0], 0, 1)
    def train(s, x, y):
        z = s.W1 @ x + s.b1; h = np.maximum(0, z); yh = float(s.W2[0] @ h + s.b2[0]); e = yh - y
        gW2 = e * h[None, :] + s.lam * s.W2; gb2 = np.array([e])
        d = (s.W2[0] * e) * (z > 0); gW1 = np.outer(d, x) + s.lam * s.W1; gb1 = d
        s.t += 1; ps = (s.W1, s.b1, s.W2, s.b2); gs = (gW1, gb1, gW2, gb2)
        for i, (p, g) in enumerate(zip(ps, gs)):
            s.m[i] = 0.9 * s.m[i] + 0.1 * g; s.v[i] = 0.999 * s.v[i] + 0.001 * g * g
            mh = s.m[i] / (1 - 0.9 ** s.t); vh = s.v[i] / (1 - 0.999 ** s.t)
            p -= s.lr * mh / (np.sqrt(vh) + 1e-8)


def heuristic(F):
    """Domain heuristic H(c) used during cold start; F: (4,11) feature rows."""
    rssi = F[:, 10] * 60 - 90
    return np.clip(0.35 + 0.012 * (rssi + 70) - 0.07 * (F[:, 0] * 6 + 0.5 * F[:, 1] * 3) + 0.05 * F[:, 4] - 0.03 * F[:, 5], 0, 1)


class AIZen(_Base):
    def __init__(s, gamma=0.99, G0=25.0, kappa=4.0, dwell=4, use_nn=True, use_gate=True, use_anom=True,
                 use_meta=True, use_safety=True, mbb=True, discount=True, seed=0, Gmin=1.0, Gmax=30.0, w=(0.60, 0.25, 0.15), nn_lr=0.01):
        super().__init__(dwell, seed)
        s.gamma, s.G0, s.kappa = gamma, G0, kappa
        s.use_nn, s.use_gate, s.use_anom, s.use_meta, s.use_safety = use_nn, use_gate, use_anom, use_meta, use_safety
        s.discount, s.Gmin, s.Gmax, s.w, s.nn_lr = discount, Gmin, Gmax, w, nn_lr
        s.ho_kind = 'mbb' if mbb else 'bbm'
    def _init(s):
        s.a = np.ones(s.K); s.b = np.ones(s.K); s.G = s.G0; s.N = 0
        s.nn = NN(11, 16, np.random.default_rng(s.seed + 5), lr=s.nn_lr)
        s.mu = None; s.var = 0.02 ** 2; s.anom = False
        s.hist = deque(maxlen=3); s.pend = None; s.lastT = 1e9; s.lastP = 0.0
    def _score(s, t):
        F = s.env['feat'][t]
        H = heuristic(F)
        if not s.use_nn: return H
        wgt = min(1.0, s.N / 40.0)
        return (1 - wgt) * H + wgt * s.nn.predict(F)
    def act(s, t, cur):
        sc = s._score(t)
        if s.discount:
            s.a = np.maximum(1.0, s.gamma * s.a); s.b = np.maximum(1.0, s.gamma * s.b)
        ae = s.a + s.kappa * sc; be = s.b + s.kappa * (1 - sc)
        th = s.rng.beta(ae, be); cand = int(np.argmax(th))
        if cur is None: s.last = t; return cand
        if cand == cur: return cur
        est = 100 * (th[cand] - th[cur]) / max(th[cur], 0.05)
        G = (s.G * (0.5 if (s.use_anom and s.anom) else 1.0)) if s.use_gate else 0.0
        if est <= G or t - s.last < s.dwell: return cur
        if s.use_safety:
            degraded = s.lastT < 5.0 or s.lastP > 10.0
            if (s.env['call'][t] or s.env['bulk'][t]) and not degraded: return cur
        s.pend = (t, float(np.mean(s.hist)) if len(s.hist) else s.lastT, [])
        s.last = t; return cand
    def aborted(s, t): s.pend = None
    def observe(s, t, arm, T, L, P, x):
        tc = s.env['P']['tcap']; r = reward(T, L, P, tc, s.w)
        s.lastT, s.lastP = T, P
        s.a[arm] += r; s.b[arm] += 1 - r; s.N += 1
        if s.use_nn: s.nn.train(x, r)
        # EWMA anomaly detector on throughput of the current link
        v = T / tc
        if s.mu is None: s.mu = v
        else:
            d = v - s.mu; sd = max(np.sqrt(s.var), 0.02)
            s.anom = abs(d) / sd > 2.5
            s.mu += 0.12 * d; s.var = 0.88 * (s.var + 0.12 * d * d)
        s.hist.append(T)
        if s.pend is not None and s.use_meta:
            s.pend[2].append(T)
            if len(s.pend[2]) >= 3:
                pre = s.pend[1]; post = float(np.mean(s.pend[2]))
                dl = 100 * (post - pre) / max(pre, 0.05 * tc)
                if dl < -5: s.G = min(s.Gmax, s.G * (1 + max(0.25, abs(dl) / 100)))
                elif dl > 40: s.G = max(s.Gmin, s.G * 0.90)
                elif dl > 15: s.G = max(s.Gmin, s.G * 0.95)
                s.pend = None
        elif s.pend is not None:
            s.pend = None
        if s.anom is False: pass


# ----------------------------------------------------------------- harness
def run_assoc(env, pol, return_trace=False):
    P = env['P']; T = env['T']; dis = env['dis']
    pol.reset(env)
    arm = np.zeros(T, int); D = np.zeros(T); ho = np.zeros(T, bool)
    cur = None
    for t in range(T):
        want = pol.act(t, cur); d = 0.0
        if cur is None:
            cur = want
        elif want != cur:
            k = pol.ho_kind
            if k == 'none':
                cur = want; ho[t] = True
            elif k == 'ft':
                d = dis['ft'][t]; cur = want; ho[t] = True
            elif k == 'csa':
                d = 2.0; cur = want; ho[t] = True
            elif k == 'mbb':
                if dis['u_fail'][t] < P['p_fail']:
                    pol.aborted(t)
                else:
                    d = dis['mbb'][t] if dis['u_mbb'][t] < P['p_mbb'] else dis['bbm'][t]; cur = want; ho[t] = True
            else:
                d = dis['bbm'][t]; cur = want; ho[t] = True
        arm[t] = cur; D[t] = min(d, DT)
        Tm = env['Ttrue'][t, cur] * np.exp(env['zT'][t, cur]) * (1 - D[t] / DT)
        Lm = float(env['rtt_samp'][t, cur].mean()); Pm = 100.0 * float(env['loss_samp'][t, cur].mean())
        pol.observe(t, cur, Tm, Lm, Pm, env['feat'][t, cur])
    return metrics(env, arm, D, ho, return_trace)


def metrics(env, arm, D, ho, trace=False):
    T = env['T']; idx = np.arange(T); P = env['P']
    Teff = env['Ttrue'][idx, arm] * (1 - D / DT)
    samp = env['rtt_samp'][idx, arm, :]
    loss = env['loss_samp'][idx, arm, :].mean(axis=1) * 100 + 100 * D / DT
    hours = T * DT / 3600.0
    Tor = env['Ttrue'].max(1)
    rew = np.array([reward(a, b, c, P['tcap']) for a, b, c in zip(Teff[::20], env['rtt_mean'][idx, arm][::20], (env['ploss'][idx, arm] * 100)[::20])]).mean()
    callho = (ho & (env['call'] | env['bulk']) & (D > 0.15)).sum()
    m = dict(tput=float(Teff.mean()), tput_p5=float(np.percentile(Teff, 5)), rtt_p50=float(np.percentile(samp, 50)),
             rtt_p99=float(np.percentile(samp, 99)), jitter=float(samp.std(axis=1).mean()), loss=float(loss.mean()),
             ho_per_h=float(ho.sum() / hours), disr_s_day=float(D.sum() / (hours / 24)), sess_ho_day=float(callho / (hours / 24)),
             regret=float(1 - Teff.mean() / Tor.mean()), reward=float(rew))
    if trace:
        m['_trace'] = dict(arm=arm, Teff=Teff, D=D, ho=ho)
    return m
