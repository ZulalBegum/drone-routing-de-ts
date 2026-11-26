# Experiment: DSDE (benchmark) vs DE-TS (ours), RUNS repetitions per size, same budget
# Output: table (console + results/results.csv), Wilcoxon test, Pareto plots (results/pareto_N.png)

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from hybrid import generate_instance, run, hypervolume, fronts

SIZES = [15, 30, 50, 100]
RUNS = 10
BUDGET = 20000
ALGORITHMS = {"DSDE": False, "DE-TS": True}

os.makedirs("results", exist_ok=True)
rows = []
for n in SIZES:
    p = generate_instance(n, seed=7)
    front = {name: [] for name in ALGORITHMS}
    times = {name: [] for name in ALGORITHMS}
    for name, hybrid in ALGORITHMS.items():
        for r in range(RUNS):
            t0 = time.time()
            archive = run(p, budget=BUDGET, hybrid=hybrid, seed=r)
            times[name].append(time.time() - t0)
            front[name].append(np.array([f for f, _ in archive]))
        print(f"n={n} {name} done")

    # common normalization over all results -> HV values are comparable
    all_F = np.vstack([F for name in front for F in front[name]])
    lo, hi = all_F.min(axis=0), all_F.max(axis=0)
    hv = {name: [hypervolume((F - lo) / (hi - lo)) for F in front[name]] for name in front}
    p_value = wilcoxon(hv["DE-TS"], hv["DSDE"]).pvalue

    for name in ALGORITHMS:
        rows.append([n, name, np.mean(hv[name]), np.std(hv[name]),
                     min(F[:, 0].min() for F in front[name]),
                     min(F[:, 1].min() for F in front[name]),
                     np.mean(times[name]), p_value])

    # Pareto plot: combined front of all runs of each algorithm
    plt.figure(figsize=(6, 4))
    for name, color in [("DSDE", "tab:red"), ("DE-TS", "tab:blue")]:
        F = np.vstack(front[name])
        F = F[fronts(list(F))[0]]
        F = F[np.argsort(F[:, 0])]
        plt.step(F[:, 0], F[:, 1], where="post", color=color, alpha=0.5)
        plt.scatter(F[:, 0], F[:, 1], color=color, label=name)
    plt.xlabel("f1 - setup cost")
    plt.ylabel("f2 - energy (Wh)")
    plt.title(f"{n} customers")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/pareto_{n}.png", dpi=130)
    plt.close()

header = ["n", "algorithm", "HV_mean", "HV_std", "best_f1", "best_f2", "time_s", "p_wilcoxon"]
with open("results/results.csv", "w") as file:
    file.write(",".join(header) + "\n")
    for row in rows:
        file.write(",".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in row) + "\n")

print(f"\n{'n':>4} {'algorithm':>9} {'HV_mean':>7} {'HV_std':>7} {'f1_min':>7} {'f2_min':>8} {'time':>6} {'p':>7}")
for row in rows:
    print(f"{row[0]:>4} {row[1]:>9} {row[2]:7.3f} {row[3]:7.3f} {row[4]:7.0f} {row[5]:8.1f} {row[6]:6.1f} {row[7]:7.4f}")
