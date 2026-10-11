"""Channel-selection environment: the optional router actuator retunes the home AP (AP1) on one band.
Arms are channels of that band. Neighbour BSSs keep their channels; their loads follow make_env."""
import numpy as np
from env import *


def make_chan_env(env, bd, ap_idx=0):
    P = env['P']; ex = env['extra']; Pap = ex['Pap']; L = ex['L']; rssi = ex['rssi_all']
    ch = env['ch']; band = ex['band_all']; mw = ex['mw']; vis = ex['vis']
    chans = CH24 if bd == 0 else CH5
    A = len(chans); n_bss = len(ch); T = env['T']
    me = 2 * ap_idx + bd
    M = np.zeros((A, n_bss)); Mint = np.zeros((A, n_bss)); SAME = np.zeros((A, n_bss))
    OVP = np.zeros((A, n_bss)); BANDM = np.zeros((A, n_bss)); RAW = np.zeros((A, n_bss))
    for k, c in enumerate(chans):
        for j in range(n_bss):
            if j == me or band[j] != bd:
                continue
            BANDM[k, j] = 1
            ov = overlap(c, ch[j], bd)
            if c == ch[j]: SAME[k, j] = 1
            elif ov > 0: OVP[k, j] = ov
            if ov <= 0: continue
            pj = Pap[j, me]
            RAW[k, j] = ov * 10 ** (pj / 10)                       # classic ACS interference metric
            if pj > (P['cca'] if ov >= 1.0 else P['ed']): M[k, j] = ov
            else: Mint[k, j] = ov ** 2
    X = L[:, None, :] * M[None, :, :]
    busy_ext = 1 - np.exp(np.log1p(-np.clip(X, 0, 0.95)).sum(axis=2))
    own = L[:, me][:, None]
    busy = 1 - (1 - busy_ext) * (1 - own)
    Imw = np.einsum('tj,kj->tk', L * mw, Mint)
    r_me = rssi[:, me][:, None]
    sinr = r_me - 10 * np.log10(10 ** (P['noise'] / 10) + Imw)
    rate = np.minimum(P['rmax'], 0.6 * 20 * 2 * np.log2(1 + 10 ** (sinr / 10)))
    Ttrue = P['eta'] * rate * np.maximum(0.05, 1 - busy)
    rq = np.clip(busy, 0, 0.985)
    rtt_mean = 4.0 + 10.0 * rq / (1.02 - rq)
    sig = lambda x: 1 / (1 + np.exp(-x))
    ploss = 0.001 + 0.05 * rq ** 3 + 0.25 * sig((10 - sinr) / 2.5)
    rng = np.random.default_rng(env['seed'] + 7919)
    zr = rng.normal(0, 1, (T, A, 10))
    rtt_samp = rtt_mean[:, :, None] * np.exp(0.4 * zr - 0.08)
    loss_samp = rng.random((T, A, 10)) < ploss[:, :, None]
    zT = rng.normal(0, 0.15, (T, A))
    cnt = vis @ BANDM.T
    x0 = np.minimum(1, (vis @ SAME.T) / 6.0); x1 = np.minimum(1, (vis @ OVP.T) / 3.0)
    mean_r = ((vis * rssi) @ BANDM.T) / np.maximum(cnt, 1)
    x2 = np.clip((mean_r + 95) / 65, 0, 1)
    mx = np.where((vis[:, None, :] * BANDM[None]) > 0, rssi[:, None, :], -120).max(axis=2)
    x3 = np.clip((mx + 95) / 65, 0, 1)
    x4 = np.array([(c in (1, 6, 11)) if bd == 0 else (c not in DFS5) for c in chans], float)
    x5 = np.array([(bd == 1 and c in DFS5) for c in chans], float)
    hr = env['hr']
    x6 = 0.5 + 0.5 * np.sin(2 * np.pi * hr / 24); x7 = 0.5 + 0.5 * np.cos(2 * np.pi * hr / 24)
    x8 = np.minimum(1, cnt / 30.0)
    x9 = np.clip((10 * np.log10((vis * mw) @ BANDM.T + 1e-12) + 100) / 60, 0, 1)
    x10 = np.broadcast_to(np.clip((r_me + 90) / 60, 0, 1), (T, A))
    feat = np.stack([x0, x1, x2, x3, np.broadcast_to(x4, (T, A)), np.broadcast_to(x5, (T, A)),
                     np.broadcast_to(x6[:, None], (T, A)), np.broadcast_to(x7[:, None], (T, A)), x8, x9, x10], axis=2)
    init = int(np.where(chans == ch[me])[0][0])
    return dict(P=P, T=T, seed=env['seed'], Ttrue=Ttrue, rtt_mean=rtt_mean, ploss=ploss, rtt_samp=rtt_samp,
                loss_samp=loss_samp, zT=zT, feat=feat, busy=busy, busy_ext=busy_ext, call=env['call'], bulk=env['bulk'],
                dis=env['dis'], chans=chans, init=init, RAW=RAW, hr=hr, rssi=np.broadcast_to(r_me, (T, A)),
                band=np.full(A, bd), arms_ap=np.zeros(A, int))


# ---- channel-selection baselines
class Static:
    ho_kind = 'csa'
    def reset(s, env): s.env = env
    def act(s, t, cur): return s.env['init']
    def observe(s, *a): pass
    def aborted(s, t): pass


class AutoBoot(Static):
    """Router 'auto channel' at boot: pick channel with least overlap-weighted neighbour power, once."""
    def act(s, t, cur):
        if cur is None or t == 0: return int(np.argmin(s.env['RAW'].sum(axis=1)))
        return cur


class ACS(Static):
    """Periodic auto-channel selection from a survey of measured channel utilisation (busy time)."""
    def __init__(s, period_h=6.0): s.period = int(period_h * 3600 / DT)
    def act(s, t, cur):
        if cur is None: return s.env['init']
        if t % s.period == 0 and t > 0: return int(np.argmin(s.env['busy_ext'][t]))
        return cur
