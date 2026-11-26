# Experiment: DSDE (benchmark) vs DE-TS (ours) on the article's benchmark instances
# Same settings as the article: population 20, 2000 evaluations, 8 independent runs per instance.
# Statistics as in the article: Wilcoxon rank-sum test (0.05) per instance on HV -> wins/losses/ties.
#
# Output (results/): runs.csv (every run), instances.csv (per instance), summary.csv (per size),
#                    pareto_<instance>.png (one example plot per size)
#
# Usage:  python experiment.py            (all instances listed below, uses all CPU cores)

import os
import time
from multiprocessing import Pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ranksums

from hybrid import Problem, run, hypervolume

RUNS = 8
BUDGET = 2000
CONFIGS = ["2,3", "3,5", "6,4"]
INSTANCES = ([f"Ca{k}-{c},15" for k in range(1, 6) for c in CONFIGS] +     # all 15 small instances
             [f"Ca{k}-{c},30" for k in range(1, 6) for c in CONFIGS] +     # all 15 medium instances
             [f"Ca{k}-3,5,50" for k in range(1, 6)] +                      # 5 large instances
             [f"Ca{k}-3,5,100" for k in range(1, 6)])                      # 5 very large instances
ALGORITHMS = {"DSDE": False, "DE-TS": True}


def one_run(job):
    name, alg, seed = job
    p = Problem(f"data/{name}.csv")
    t0 = time.time()
    archive = run(p, budget=BUDGET, hybrid=ALGORITHMS[alg], seed=seed)
    F = np.array([f for f, _ in archive]) if archive else np.empty((0, 2))
    return name, alg, seed, F, time.time() - t0


def nd(F):
    return F[[i for i in range(len(F)) if not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i])
                                                  for j in range(len(F)))]]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    jobs = [(name, alg, s) for name in INSTANCES for alg in ALGORITHMS for s in range(RUNS)]
    jobs.sort(key=lambda j: -int(j[0].split(",")[-1]))           # large instances first
    out = {}
    with Pool(os.cpu_count()) as pool:
        for k, (name, alg, seed, F, sec) in enumerate(pool.imap_unordered(one_run, jobs), 1):
            out[(name, alg, seed)] = (F, sec)
            if k % 20 == 0 or k == len(jobs):
                print(f"{k}/{len(jobs)} runs done", flush=True)

    run_rows, inst_rows = [], []
    for name in INSTANCES:
        allF = np.vstack([out[(name, a, s)][0] for a in ALGORITHMS for s in range(RUNS) if len(out[(name, a, s)][0])])
        lo, hi = allF.min(axis=0), allF.max(axis=0)
        hv = {a: [hypervolume((out[(name, a, s)][0] - lo) / np.where(hi > lo, hi - lo, 1)) for s in range(RUNS)]
              for a in ALGORITHMS}
        for a in ALGORITHMS:
            for s in range(RUNS):
                F, sec = out[(name, a, s)]
                run_rows.append([name, a, s, hv[a][s], len(F), sec])
        pval = ranksums(hv["DE-TS"], hv["DSDE"]).pvalue
        verdict = "=" if pval >= 0.05 else ("+" if np.mean(hv["DE-TS"]) > np.mean(hv["DSDE"]) else "-")
        best = {a: nd(np.vstack([out[(name, a, s)][0] for s in range(RUNS)])) for a in ALGORITHMS}
        inst_rows.append([name, int(name.split(",")[-1]), np.mean(hv["DSDE"]), np.std(hv["DSDE"]),
                          np.mean(hv["DE-TS"]), np.std(hv["DE-TS"]), pval, verdict,
                          best["DSDE"][:, 1].min(), best["DE-TS"][:, 1].min(),
                          np.mean([out[(name, "DSDE", s)][1] for s in range(RUNS)]),
                          np.mean([out[(name, "DE-TS", s)][1] for s in range(RUNS)])])

        if name.startswith("Ca1-3,5") or name == "Ca1-2,3,15":   # one example plot per size
            plt.figure(figsize=(6, 4))
            for a, color in [("DSDE", "tab:red"), ("DE-TS", "tab:blue")]:
                B = best[a][np.argsort(best[a][:, 0])]
                plt.step(B[:, 0], B[:, 1], where="post", color=color, alpha=0.5)
                plt.scatter(B[:, 0], B[:, 1], color=color, label=a)
            plt.xlabel("f1 - dispatch cost ($)")
            plt.ylabel("f2 - operational cost ($)")
            plt.title(f"{name}: combined front of {RUNS} runs")
            plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
            plt.savefig(f"results/pareto_{name}.png", dpi=130); plt.close()

    def write(path, header, rows):
        with open(path, "w") as f:
            f.write(",".join(header) + "\n")
            for r in rows:
                f.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + "\n")

    write("results/runs.csv", ["instance", "algorithm", "run", "HV", "front_size", "time_s"], run_rows)
    write("results/instances.csv", ["instance", "n", "HV_DSDE", "std_DSDE", "HV_DETS", "std_DETS", "p_ranksum",
                                    "DETS_vs_DSDE", "best_f2_DSDE", "best_f2_DETS", "time_DSDE", "time_DETS"],
          inst_rows)

    summary = []
    for n in sorted({r[1] for r in inst_rows}):
        rs = [r for r in inst_rows if r[1] == n]
        w = sum(r[7] == "+" for r in rs); l = sum(r[7] == "-" for r in rs); t = sum(r[7] == "=" for r in rs)
        summary.append([n, len(rs), np.mean([r[2] for r in rs]), np.mean([r[4] for r in rs]), f"{w}/{l}/{t}",
                        np.mean([r[10] for r in rs]), np.mean([r[11] for r in rs])])
    write("results/summary.csv", ["n", "instances", "HV_DSDE", "HV_DETS", "DETS_vs_DSDE(+/-/=)",
                                  "time_DSDE", "time_DETS"], summary)
    print(f"\n{'n':>4} {'inst':>4} {'HV DSDE':>8} {'HV DE-TS':>9} {'+/-/=':>7} {'t DSDE':>7} {'t DE-TS':>8}")
    for r in summary:
        print(f"{r[0]:>4} {r[1]:>4} {r[2]:8.3f} {r[3]:9.3f} {r[4]:>7} {r[5]:7.1f} {r[6]:8.1f}")
