# DE + Tabu Search Hybrid – Drone Station Location and Routing

ENS001 – Application Development for Optimization

A bi-objective optimization application that decides **where to open drone stations**, **how many drones to use**
and **in which order each drone visits its customers**, while minimizing:

- **f1 – setup cost** = open stations × 1000 + drones × 200
- **f2 – total energy (Wh)**, where arc energy = distance × (20 + 4 × carried load)

Constraints: route load ≤ 5 kg, route energy ≤ 400 Wh. Parameters are at the top of `hybrid.py`.

## Files

| File | Purpose |
|---|---|
| `hybrid.py` | Problem, solution decoding, DE, Tabu Search, Pareto tools. `run()` does everything. |
| `experiment.py` | Runs DSDE (benchmark) and DE-TS (ours) on 15/30/50/100 customers, 10 runs each; writes `results/`. |
| `app.py` | Streamlit user interface. |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python hybrid.py         # quick test (30 customers)
python experiment.py     # experiments for the report -> results/
streamlit run app.py     # user interface
```

From your own code:

```python
from hybrid import generate_instance, run

p = generate_instance(30, seed=7)          # or Problem(customers, demand, stations)
archive = run(p, budget=20000, hybrid=True)
for f, solution in archive:                # f = [f1, f2], solution = [[station, [customer order]], ...]
    print(f, solution)
```

## Method

- **Differential Evolution (DE):** 20 solution plans; mutation using neighbors in decision space (niching) + crossover;
  selection by non-dominated sorting + crowding distance in decision space. Explores setup cost (f1).
- **Tabu Search:** every 5 generations, applied to the two extremes of the archive + one random solution.
  Moves: swap order, reverse segment, exchange, relocate, merge. Reduces energy (f2).
- **Tabu list:** a pair of customers that was just swapped is forbidden for a few iterations (prevents cycling).
- **Two objectives:** neighbors are compared with **Pareto dominance** instead of a single cost.
- **DSDE (benchmark):** the same DE without Tabu Search -> `run(p, hybrid=False)`.

## Results (random instances, 10 runs, 20 000 evaluations)

| Customers | DSDE HV | DE-TS HV | Wilcoxon p |
|---|---|---|---|
| 15  | 0.626 | **0.786** | 0.002 |
| 30  | 0.763 | **0.998** | 0.002 |
| 50  | 0.724 | **0.937** | 0.002 |
| 100 | 0.804 | **0.982** | 0.002 |

Higher hypervolume (HV) is better.
