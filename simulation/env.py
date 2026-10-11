"""Wi-Fi apartment-building environment for evaluating client-side network selection.

All quantities are *modelled* (not measured). Parameters are assumptions in ranges
commonly used for indoor Wi-Fi, listed in DEFAULT and varied in the sensitivity study.
"""
import numpy as np

DT = 30.0                       # decision / scan interval [s]
CH24 = np.arange(1, 12)
CH5 = np.array([36, 40, 44, 48, 52, 56, 60, 64, 149, 153, 157, 161])
DFS5 = {52, 56, 60, 64}

DEFAULT = dict(days=3, n_nb=19, cell=12.0, load_scale=3.0, n24=3.0, n5=3.4,
               W_unit24=7.0, W_unit5=9.0, W_int=3.0, sigma_sh=4.0, eirp=20.0,
               noise=-95.0, eta=0.65, tcap=180.0, p_mbb=0.9, p_fail=0.03,
               cca=-82.0, ed=-62.0, rmax=288.0, d_bbm=0.45, traffic='markov')

PL1M = (40.2, 46.8)             # free-space loss at 1 m for 2.4 / 5 GHz


def _cells(p):
    c = p['cell']
    return [((i % 7) * c + c / 2, (i // 7) * c + c / 2) for i in range(21)]


def pathloss(p1, p2, band, P):
    """log-distance + unit-wall + interior-partition loss. p1,p2: (...,2)."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    d = np.maximum(np.linalg.norm(p1 - p2, axis=-1), 1.0)
    n = P['n24'] if band == 0 else P['n5']
    W = P['W_unit24'] if band == 0 else P['W_unit5']
    cu = P['cell']
    unit = (np.abs(np.floor(p1[..., 0] / cu) - np.floor(p2[..., 0] / cu)) +
            np.abs(np.floor(p1[..., 1] / cu) - np.floor(p2[..., 1] / cu)))
    g = 6.0
    tot = (np.abs(np.floor(p1[..., 0] / g) - np.floor(p2[..., 0] / g)) +
           np.abs(np.floor(p1[..., 1] / g) - np.floor(p2[..., 1] / g)))
    interior = np.maximum(tot - unit, 0)
    return PL1M[band] + 10 * n * np.log10(d) + W * unit + P['W_int'] * interior


def draw_channel(rng, band):
    if band == 0:
        if rng.random() < 0.7:
            return int(rng.choice([1, 6, 11]))
        return int(rng.integers(1, 12))
    w = np.array([1 if c in DFS5 else 3 for c in CH5], float)
    return int(rng.choice(CH5, p=w / w.sum()))


def overlap(ca, cb, band):
    if band == 0:
        return max(0.0, 1 - abs(ca - cb) / 5.0)
    return 1.0 if ca == cb else 0.0


def diurnal(t_steps):
    h = (np.arange(t_steps) * DT / 3600.0) % 24
    return 0.35 + 0.65 * np.exp(-0.5 * (((h - 21 + 12) % 24 - 12) / 3.0) ** 2)


def make_world(seed, **over):
    P = dict(DEFAULT); P.update(over)
    rng = np.random.default_rng(seed)
    cells = _cells(P)
    home = 10
    nb = [i for i in range(21) if i != home]
    rng.shuffle(nb)
    nb = nb[:P['n_nb']]
    hx, hy = cells[home][0] - P['cell'] / 2, cells[home][1] - P['cell'] / 2
    aps = [(hx + 2.0, hy + 6.0), (hx + 10.0, hy + 6.0)]
    for i in nb:
        cx, cy = cells[i]
        aps.append((cx + rng.uniform(-4, 4), cy + rng.uniform(-4, 4)))
    aps = np.array(aps)
    n_ap = len(aps)
    # BSS j = 2*ap + band
    n_bss = 2 * n_ap
    ch = np.zeros(n_bss, int)
    for j in range(n_bss):
        ch[j] = draw_channel(rng, j % 2)
    spots = np.array([(hx + 1.0, hy + 2.0), (hx + 6.0, hy + 10.0), (hx + 11.0, hy + 10.0)])
    spots = spots + rng.uniform(-0.5, 0.5, spots.shape)
    return dict(P=P, rng=rng, aps=aps, n_ap=n_ap, n_bss=n_bss, ch=ch, spots=spots)


def make_env(seed, **over):
    """Association-selection environment: arms = home BSSIDs (AP1-2.4, AP1-5, AP2-2.4, AP2-5)."""
    W = make_world(seed, **over)
    P, rng = W['P'], W['rng']
    aps, n_bss, ch, spots = W['aps'], W['n_bss'], W['ch'], W['spots']
    T = int(P['days'] * 86400 / DT)
    band = np.arange(n_bss) % 2
    ap_of = np.arange(n_bss) // 2
    arms = [0, 1, 2, 3]                       # BSS indices of the home BSSIDs
    # --- static AP-AP received power (per band), symmetric shadowing
    Pap = np.zeros((n_bss, n_bss))
    sh_static = rng.normal(0, P['sigma_sh'], (len(aps), len(aps)))
    sh_static = (sh_static + sh_static.T) / 2
    for a in range(len(aps)):
        for b in range(len(aps)):
            if a == b:
                continue
            for bd in (0, 1):
                Pap[2 * a + bd, 2 * b + bd] = P['eirp'] - pathloss(aps[a], aps[b], bd, P) + sh_static[a, b]
    # --- coupling matrices for the 4 home arms
    M = np.zeros((4, n_bss)); Mint = np.zeros((4, n_bss))
    SAME = np.zeros((4, n_bss)); OVP = np.zeros((4, n_bss)); BANDM = np.zeros((4, n_bss))
    for k, b in enumerate(arms):
        bd = band[b]; a = ap_of[b]
        for j in range(n_bss):
            if j == b or band[j] != bd:
                continue
            BANDM[k, j] = 1
            ov = overlap(ch[b], ch[j], bd)
            if ch[b] == ch[j]:
                SAME[k, j] = 1
            elif ov > 0:
                OVP[k, j] = ov
            if ov <= 0:
                continue
            pj = Pap[j, b] if ap_of[j] != a else P['eirp'] - 10
            thr = P['cca'] if ov >= 1.0 else P['ed']
            if pj > thr:
                M[k, j] = ov
            else:
                Mint[k, j] = ov ** 2
    # --- client positions over time (3 spots, mean dwell 3 h)
    pos = np.zeros(T, int)
    cur = int(rng.integers(0, 3))
    stay = 1.0 / (3 * 3600 / DT)
    mv = rng.random(T)
    for t in range(T):
        if mv[t] < stay:
            cur = int((cur + rng.integers(1, 3)) % 3)
        pos[t] = cur
    # --- static client PL for each spot and BSS
    PLc = np.zeros((3, n_bss))
    for s in range(3):
        for j in range(n_bss):
            PLc[s, j] = pathloss(spots[s], aps[ap_of[j]], band[j], P)
    # --- AR(1) shadowing per BSS
    rho = 0.995
    eps = rng.normal(0, 1, (T, n_bss))
    sh = np.zeros((T, n_bss))
    sh[0] = eps[0] * P['sigma_sh']
    for t in range(1, T):
        sh[t] = rho * sh[t - 1] + np.sqrt(1 - rho ** 2) * P['sigma_sh'] * eps[t]
    rssi = P['eirp'] - PLc[pos] + sh                       # (T, n_bss)
    # --- traffic load processes
    dfac = diurnal(T)
    levels = np.array([0.1, 0.25, 0.5]) * P['load_scale']
    ell = rng.choice(levels, size=n_bss, p=[0.5, 0.35, 0.15])
    # the two bands of one AP share the same household -> correlated activity
    u_on = rng.random((T, len(aps))); u_off = rng.random((T, len(aps)))
    act = np.zeros((T, len(aps)), bool)
    st = rng.random(len(aps)) < 0.2
    for t in range(T):
        p_on = 0.02 * dfac[t]
        st = np.where(st, u_off[t] > 0.04, u_on[t] < p_on)
        act[t] = st
    if P['traffic'] == 'pareto':       # heavy-tailed ON/OFF durations, same mean activity as the Markov model
        r2 = np.random.default_rng(seed + 555); alpha_p = 1.5
        for a in range(len(aps)):
            t = 0; on = bool(r2.random() < 0.2); col = np.zeros(T, bool)
            while t < T:
                mean = 25.0 if on else 1.0 / (0.02 * dfac[min(t, T - 1)])
                d = int(max(1, (mean * (alpha_p - 1) / alpha_p) * (1 - r2.random()) ** (-1 / alpha_p)))
                col[t:t + d] = on; t += d; on = not on
            act[:, a] = col
    # per-AP band split: active household uses 5 GHz with prob .6 / 2.4 with .4 (static per AP)
    split = rng.random(len(aps)) < 0.6
    L = np.full((T, n_bss), 0.02)
    for j in range(n_bss):
        a = ap_of[j]
        uses = (split[a] if band[j] == 1 else ~split[a])
        L[:, j] += act[:, a] * ell[j] * (1.0 if uses else 0.35)
    L = np.clip(L, 0, 0.95)
    # --- per-arm channel state
    X = L[:, None, :] * M[None, :, :]                      # (T,4,n_bss)
    busy_ext = 1 - np.exp(np.log1p(-np.clip(X, 0, 0.95)).sum(axis=2))
    own = L[:, arms]                                       # other-device load on own BSS
    busy_tot = 1 - (1 - busy_ext) * (1 - own)
    mw = 10 ** (rssi / 10)
    Imw = np.einsum('tj,kj->tk', L * mw, Mint)
    nmw = 10 ** (P['noise'] / 10)
    rssi_arm = rssi[:, arms]
    sinr = rssi_arm - 10 * np.log10(nmw + Imw)
    rate = np.minimum(P['rmax'], 0.6 * 20 * 2 * np.log2(1 + 10 ** (sinr / 10)))
    air = np.maximum(0.05, 1 - busy_tot)
    Ttrue = P['eta'] * rate * air
    rho_q = np.clip(busy_tot, 0, 0.985)
    rtt_mean = 4.0 + 10.0 * rho_q / (1.02 - rho_q)
    sig = lambda x: 1 / (1 + np.exp(-x))
    ploss = 0.001 + 0.05 * rho_q ** 3 + 0.25 * sig((10 - sinr) / 2.5)
    # --- measurement samples (common random numbers across policies)
    zr = rng.normal(0, 1, (T, 4, 10))
    rtt_samp = rtt_mean[:, :, None] * np.exp(0.4 * zr - 0.08)
    loss_samp = rng.random((T, 4, 10)) < ploss[:, :, None]
    zT = rng.normal(0, 0.15, (T, 4))
    # --- scan features for the neural scorer (RF-only; loads are NOT observable)
    vis = (rssi > -90).astype(float)
    cnt = vis @ BANDM.T
    x0 = np.minimum(1, (vis @ SAME.T) / 6.0)
    x1 = np.minimum(1, (vis @ OVP.T) / 3.0)
    mean_r = ((vis * rssi) @ BANDM.T) / np.maximum(cnt, 1)
    x2 = np.clip((mean_r + 95) / 65, 0, 1)
    mx = np.where((vis[:, None, :] * BANDM[None]) > 0, rssi[:, None, :], -120).max(axis=2)
    x3 = np.clip((mx + 95) / 65, 0, 1)
    chk = np.array([ch[b] for b in arms]); bdk = band[arms]
    x4 = np.array([(c in (1, 6, 11)) if b == 0 else (c not in DFS5) for c, b in zip(chk, bdk)], float)
    x5 = np.array([(b == 1 and c in DFS5) for c, b in zip(chk, bdk)], float)
    hr = (np.arange(T) * DT / 3600.0) % 24
    x6 = 0.5 + 0.5 * np.sin(2 * np.pi * hr / 24); x7 = 0.5 + 0.5 * np.cos(2 * np.pi * hr / 24)
    x8 = np.minimum(1, cnt / 30.0)
    tot = ((vis * mw) @ BANDM.T)
    x9 = np.clip((10 * np.log10(tot + 1e-12) + 100) / 60, 0, 1)
    x10 = np.clip((rssi_arm + 90) / 60, 0, 1)
    feat = np.stack([x0, x1, x2, x3,
                     np.broadcast_to(x4, (T, 4)), np.broadcast_to(x5, (T, 4)),
                     np.broadcast_to(x6[:, None], (T, 4)), np.broadcast_to(x7[:, None], (T, 4)),
                     x8, x9, x10], axis=2)
    # --- sessions and disruption draws
    call = np.zeros(T, bool); bulk = np.zeros(T, bool)
    uc = rng.random((T, 2)); cs = False; bs = False
    for t in range(T):
        cs = (uc[t, 0] > 0.04) if cs else (uc[t, 0] < 0.0015 * dfac[t] * 2)
        bs = (uc[t, 1] > 0.02) if bs else (uc[t, 1] < 0.002)
        call[t] = cs; bulk[t] = bs
    dis = dict(bbm=np.exp(np.log(P['d_bbm']) + 0.6 * rng.normal(size=T)),
               ft=np.exp(np.log(0.08) + 0.4 * rng.normal(size=T)),
               mbb=np.exp(np.log(0.025) + 0.4 * rng.normal(size=T)),
               u_mbb=rng.random(T), u_fail=rng.random(T), u_acc=rng.random(T))
    return dict(P=P, T=T, seed=seed, rssi=rssi_arm, Ttrue=Ttrue, rtt_mean=rtt_mean, ploss=ploss,
                rtt_samp=rtt_samp, loss_samp=loss_samp, zT=zT, feat=feat, busy=busy_tot,
                pos=pos, call=call, bulk=bulk, dis=dis, band=np.array([0, 1, 0, 1]),
                ch_arms=chk, hr=hr, sinr=sinr, rate=rate, air=air, aps=aps, ch=ch, arms_ap=np.array([0, 0, 1, 1]),
                extra=dict(Pap=Pap, L=L, rssi_all=rssi, ap_of=ap_of, band_all=band, mw=mw, vis=vis, spots=spots, pos=pos))
