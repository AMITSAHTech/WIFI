Wi-Fi AI Zen simulator (supplementary material)
Run order: tune.py -> exp1.py -> exp2.py -> exp3.py -> exp4.py -> exp5.py ; figures: figs_exp1/3/4.py, figs_static.py.
env.py / env2.py : apartment-building environment (association and channel modes)
policies.py      : Wi-Fi AI Zen, ablations, baselines, evaluation harness
runlib.py        : policy constructors using tuned.json (hyperparameters selected on validation seeds 100-105 / 200-205)
All numbers in the paper come from the *.pkl result files produced by these scripts. Results are simulation-based.
