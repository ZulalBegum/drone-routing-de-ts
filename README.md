# DE + Tabu Search Hybrid for Two-Echelon Truck–Drone Routing (2E-VRPLDTW)

ENS001 – Application Development for Optimization

This project solves the problem of the article

> F. Yu, C. Li, Q. Chen, Y. Li, Y. Chen, X. Yang (2027). *Multi-objective two-echelon vehicle-drone routing with
> load-dependent drone under pickup delivery time windows.* Transportation Research Part E, 217, 105195.

and compares the article's algorithm (**DSDE**) with our hybrid (**DE-TS = DSDE + Pareto Tabu Search**).

## The problem

- **1st echelon:** trucks carry delivery parcels from a depot to satellites.
- **2nd echelon:** drones leave a satellite, serve customers inside their **time windows**, and return to the
  satellite **nearest to their last customer**. Half of the customers are **deliveries**, half are **pickups**.
- Drone power depends on the carried payload: `P = (5 + load)^1.5 · C` (W). If a drone arrives early it
  **hovers** and keeps using energy.

| | |
|---|---|
| **f1 – dispatch cost** | 12.5 $ per truck + 3.36 $ per drone trip |
| **f2 – operational cost** | 0.1286 $/km for trucks + 0.001 $/Wh for drones |
| Hard constraints | drone payload ≤ 10 kg, drone energy ≤ 550 Wh, truck capacity 100 kg |
| Time windows | must be met in the final solutions (violation = total lateness, minutes) |

Parameters follow the article and the authors' public code (https://github.com/Yu970623/DSDE-2EVRPLDTW).
**Model check:** for instance Ca1-2,3,15 our evaluation reproduces the objective values of the authors'
reference solution exactly, (25.94, 1.611), which is also the value in the article's Table 6
(see `tests/test_model.py`).

## Why a hybrid?

The article's Table 7 shows that DSDE beats local-search heuristics on large instances but **loses to LNS and
ALNS on small instances** (15 customers: LNS better on 11/15, ALNS on 12/15). Our idea: DSDE keeps the
population-level view (which satellites, how many drones → f1), and a **memory-based Pareto Tabu Search**
fine-tunes the routes (→ f2).

## Method

**DSDE (benchmark, our re-implementation of the article's algorithm)**
- Random-key encoding: satellite open keys, customer→satellite keys, visiting priority keys, insertion bias gene.
- Decoding by cheapest feasible insertion (payload and battery are always respected).
- DE/rand/1 with CR = 1, F = 0.5, plus the article's elite-guided / elite-attracted hybrid operator.
- Dynamic constraint tolerance ε for time-window violations (ε → 0 after 80 % of the budget).
- Environmental selection with decision-space niche constrained dominance (DSDN-CDP) and a dynamic niche radius.

**DE-TS (ours)** = DSDE + every 5 generations a Pareto Tabu Search on the archive's two extreme solutions
(lowest f1, lowest f2):
- Moves: swap order in a trip, relocate a customer, exchange customers, merge two trips, give a customer its own
  drone, launch a trip from another satellite.
- **Tabu list:** recently moved customers (pairs) are forbidden for a few iterations (no cycling).
- **Two objectives:** neighbours are compared with **constrained Pareto dominance**, not one weighted cost.
- At most 50 % of the evaluation budget goes to Tabu Search; improved solutions are written back into the
  DE population.

Both algorithms use the article's settings: population 20, **2000 evaluations**, 8 runs per instance.

## Files

| File | Purpose |
|---|---|
| `hybrid.py` | Model, decoder, DSDE, Tabu Search, DE-TS. `run(p, hybrid=True/False)` |
| `experiment.py` | Benchmark experiment → `results/` |
| `app.py` | Streamlit user interface |
| `data/` | The article's 60 benchmark instances (Ca1–Ca5) |
| `tests/` | Model and algorithm tests |

## Usage

```bash
pip install -r requirements.txt
python hybrid.py "data/Ca1-2,3,15.csv"   # quick run on one instance
python experiment.py                     # experiments for the report -> results/
streamlit run app.py                     # user interface
python -m pytest -q                      # tests
```

From your own code:

```python
from hybrid import Problem, run, details

p = Problem("data/Ca1-2,3,30.csv")
archive = run(p, budget=2000, hybrid=True)     # [(f, solution), ...], solution = [[satellite, [customers]], ...]
info = details(p, archive[0][1])               # truck routes, drone trips, energy, payload, lateness
```

## Results

Article settings (population 20, 2000 evaluations, 8 runs). HV is normalized per instance over all runs of both
algorithms (reference point 1.1, higher is better). "+/-/=" = number of instances where DE-TS is significantly
better / worse / not different (Wilcoxon rank-sum test, 0.05), the same test the article uses.

| Customers | Instances | HV DSDE | HV DE-TS | DE-TS vs DSDE (+/-/=) | Time DSDE (s) | Time DE-TS (s) |
|---|---|---|---|---|---|---|
| 15  | 15 | 0.790 | **0.841** | **6/0/9** | 1.7  | 0.8  |
| 30  | 15 | 0.877 | 0.866 | 0/1/14 | 5.0  | 2.3  |
| 50  | 5  | 0.891 | 0.832 | 0/1/4  | 11.5 | 5.1  |
| 100 | 5  | 0.913 | 0.867 | 0/1/4  | 42.5 | 18.1 |

- On **small instances** – exactly where the article reports DSDE losing to local-search methods – the Tabu
  Search gives a clear improvement (better on 6 of 15 instances, never worse).
- On **30 customers** both methods are statistically equal (14 of 15 ties).
- On **50/100 customers** most differences are not significant, but the average HV of DE-TS is lower: Tabu
  Search uses half of the budget, so DE gets fewer generations, and on large instances the population-level
  search matters more. A size-dependent Tabu budget (`ls_ratio`) is the natural next step.
- DE-TS needs about **half the running time**, because a Tabu move is evaluated without re-decoding.

Per-instance numbers: `results/instances.csv`; every run: `results/runs.csv`.
