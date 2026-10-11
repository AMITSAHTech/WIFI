import itertools, json, time, sys, os, numpy as np
from env import *
from policies import *
from concurrent.futures import ProcessPoolExecutor, as_completed

VAL = list(range(100, 106))
DAYS = 2

# Worker-level cached environments
_worker_envs = None

def _init_worker(days, val_seeds):
    global _worker_envs
    _worker_envs = [make_env(s, days=days) for s in val_seeds]

def _eval_combo_worker(args):
    name, kw = args
    global _worker_envs
    if name == 'EpsSticky':
        mk = lambda seed: EpsSticky(seed=seed, **kw)
    elif name == 'DUCB':
        mk = lambda seed: DUCB(seed=seed, **kw)
    elif name == 'SWUCB':
        mk = lambda seed: SWUCB(seed=seed, **kw)
    elif name == 'TSStat':
        mk = lambda seed: TSStat(seed=seed, **kw)
    elif name == 'SWTS':
        mk = lambda seed: SWTS(seed=seed, **kw)
    elif name == 'AIZen':
        mk = lambda seed: AIZen(seed=seed, **kw)
    else:
        raise ValueError(f"Unknown policy {name}")
    r = np.mean([run_assoc(e, mk(i))['reward'] for i, e in enumerate(_worker_envs)])
    return float(r), kw

def main():
    grids = {
        'EpsSticky': (dict(eps=[0.02, 0.05, 0.1, 0.2], alpha=[0.1, 0.3], dwell=[2, 4, 8])),
        'DUCB':      (dict(gamma=[0.98, 0.99, 0.995], dwell=[2, 4, 8])),
        'SWUCB':     (dict(tau=[100, 300, 1000], dwell=[2, 4, 8])),
        'TSStat':    (dict(dwell=[2, 4, 8])),
        'SWTS':      (dict(tau=[100, 300, 1000], dwell=[2, 4, 8])),
        'AIZen':     (dict(gamma=[0.95, 0.99, 0.998], G0=[1.0, 3.0, 8.0], kappa=[2.0, 4.0, 8.0], dwell=[2, 4])),
    }

    # Load existing tuned.json if present to allow resuming
    best = {}
    log = {}
    if os.path.exists('tuned.json'):
        try:
            with open('tuned.json', 'r') as f:
                d = json.load(f)
                best = d.get('best', {})
                log = d.get('top5', {})
        except Exception:
            pass

    num_workers = min(os.cpu_count() or 4, 12)
    print(f"Starting tuning across {len(VAL)} validation seeds (days={DAYS}) using {num_workers} parallel workers...", flush=True)

    with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(DAYS, VAL)) as executor:
        for name, g in grids.items():
            keys = list(g)
            combos = [dict(zip(keys, vals)) for vals in itertools.product(*[g[k] for k in keys])]
            total = len(combos)
            res = []
            t0 = time.time()
            print(f"[{name}] tuning {total} parameter combinations...", flush=True)

            # Submit all combinations to the worker pool
            futures = [executor.submit(_eval_combo_worker, (name, kw)) for kw in combos]
            done_count = 0
            for fut in as_completed(futures):
                score, kw = fut.result()
                res.append((score, kw))
                done_count += 1
                if done_count % max(1, total // 5) == 0 or done_count == total:
                    elapsed = time.time() - t0
                    print(f"  [{name}] progress: {done_count}/{total} combos ({elapsed:.1f}s)", flush=True)

            res.sort(key=lambda x: -x[0])
            best[name] = res[0][1]
            log[name] = res[:5]
            print(f"{name} best {res[0]} ({time.time() - t0:.0f}s)\n", flush=True)

            # Checkpoint after each policy so no progress is ever lost
            with open('tuned.json', 'w') as f:
                json.dump(dict(best=best, top5=log), f, indent=1)

    print('done', flush=True)

if __name__ == '__main__':
    main()
