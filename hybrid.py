# Drone station location + routing (bi-objective)
# Method:     Differential Evolution + Pareto Tabu Search hybrid (DE-TS)
# Benchmark:  DSDE (same DE, no Tabu Search)  -> hybrid=False
#
# f1 = setup cost   = open stations * STATION_COST + number of drones * DRONE_COST
# f2 = total energy (Wh), arc energy = distance * (E0 + E1 * carried load)

import numpy as np

STATION_COST = 1000
DRONE_COST = 200
Q = 5.0      # drone payload capacity (kg)
B = 400.0    # battery capacity (Wh)
E0 = 20.0    # empty flight energy (Wh/km)
E1 = 4.0     # extra energy per kg (Wh/km/kg)


# ------------------------------------------------------------------ PROBLEM
class Problem:
    def __init__(self, customers, demand, stations):
        self.customers = np.array(customers, float)
        self.demand = np.array(demand, float)
        self.stations = np.array(stations, float)
        self.n, self.m = len(customers), len(stations)
        # distances: customer-customer and customer-station
        self.d_cc = np.linalg.norm(self.customers[:, None] - self.customers[None, :], axis=2)
        self.d_cs = np.linalg.norm(self.customers[:, None] - self.stations[None, :], axis=2)
        # can customer i be served alone from station j?
        single = self.d_cs * (2 * E0 + E1 * self.demand[:, None])
        self.reachable = single <= B
        self.nearest = np.argsort(self.d_cs, axis=1)   # stations sorted by distance, per customer


def generate_instance(n, m=None, seed=0, area=10.0):
    rng = np.random.default_rng(seed)
    m = m or max(3, n // 5)
    stations = rng.uniform(0.1 * area, 0.9 * area, (m, 2))
    customers, demand = [], []
    while len(customers) < n:
        p = rng.uniform(0, area, 2)
        q = round(rng.uniform(0.5, 2.5), 2)
        d = np.linalg.norm(stations - p, axis=1).min()
        if d * (2 * E0 + E1 * q) <= 0.9 * B:      # must be reachable from at least one station
            customers.append(p)
            demand.append(q)
    return Problem(customers, demand, stations)


# ------------------------------------------------------------------ SOLUTION
# Solution = [[station, [customer order]], ...]   (each route is one drone)

def route_energy(p, st, route):
    if not route:
        return 0.0
    load = p.demand[route].sum()
    e = p.d_cs[route[0], st] * (E0 + E1 * load)
    load -= p.demand[route[0]]
    for a, b in zip(route[:-1], route[1:]):
        e += p.d_cc[a, b] * (E0 + E1 * load)
        load -= p.demand[b]
    return e + p.d_cs[route[-1], st] * E0      # empty return flight


def route_feasible(p, st, route):
    return p.demand[route].sum() <= Q and route_energy(p, st, route) <= B


def objectives(p, solution):
    used_stations = {st for st, route in solution}
    f1 = len(used_stations) * STATION_COST + len(solution) * DRONE_COST
    f2 = sum(route_energy(p, st, route) for st, route in solution)
    return np.array([f1, f2])


def decode(p, x, k=3):
    """Turns a DE vector (random keys) into routes.
    x[:m] open/close stations, x[m:m+n] customer assignment, x[m+n:] visiting priority"""
    m, n = p.m, p.n
    is_open = x[:m] > 0.5
    if not is_open.any():
        is_open[np.argmax(x[:m])] = True
    assign = np.zeros(n, int)
    for i in range(n):
        cands = [j for j in p.nearest[i] if is_open[j] and p.reachable[i, j]]
        if not cands:                                 # repair: open nearest reachable station
            j = next(j for j in p.nearest[i] if p.reachable[i, j])
            is_open[j] = True
            cands = [j]
        cands = cands[:k]
        assign[i] = cands[min(int(x[m + i] * len(cands)), len(cands) - 1)]
    solution = []
    for st in np.unique(assign):
        ordered = sorted(np.where(assign == st)[0], key=lambda i: x[m + n + i])
        route = []
        for i in ordered:                             # same drone until full, then a new drone
            if route_feasible(p, st, route + [i]):
                route.append(i)
            else:
                solution.append([int(st), route])
                route = [i]
        solution.append([int(st), route])
    return solution


def encode(p, solution, rng, k=3):
    """Turns a Tabu-improved solution back into a DE vector (to put it back into the population)."""
    m, n = p.m, p.n
    x = np.zeros(m + 2 * n)
    used = {st for st, _ in solution}
    for j in range(m):
        x[j] = rng.uniform(0.6, 0.9) if j in used else rng.uniform(0.1, 0.4)
    pos = 0
    for st, route in solution:
        for i in route:
            cands = [j for j in p.nearest[i] if j in used and p.reachable[i, j]][:k]
            idx = cands.index(st) if st in cands else 0
            x[m + i] = (idx + 0.5) / len(cands)
            x[m + n + i] = (pos + 0.5) / n
            pos += 1
    return x


# ------------------------------------------------------------------ PARETO
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def add_to_archive(archive, f, solution):
    """archive = [(f, solution), ...] keeps only non-dominated solutions"""
    for g, _ in archive:
        if np.all(g <= f):
            return False
    archive[:] = [(g, s) for g, s in archive if not dominates(f, g)] + [(f, solution)]
    return True


def fronts(F):
    """Non-dominated sorting: [[1st front indices], [2nd front], ...]"""
    remaining = list(range(len(F)))
    result = []
    while remaining:
        front = [i for i in remaining if not any(dominates(F[j], F[i]) for j in remaining)]
        result.append(front)
        remaining = [i for i in remaining if i not in front]
    return result


def crowding(V):
    n = len(V)
    d = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for j in range(V.shape[1]):
        s = np.argsort(V[:, j])
        span = V[s[-1], j] - V[s[0], j]
        d[s[0]] = d[s[-1]] = np.inf
        if span > 0:
            d[s[1:-1]] += (V[s[2:], j] - V[s[:-2], j]) / span
    return d


def hypervolume(F, ref=(1.1, 1.1)):
    """Hypervolume of a normalized 2-objective front (higher = better)"""
    F = sorted(tuple(f) for f in F if f[0] < ref[0] and f[1] < ref[1])
    hv, prev_y = 0.0, ref[1]
    for x, y in F:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv


# ------------------------------------------------------------------ TABU SEARCH
def neighbor(p, solution, rng):
    """Random small change. Returns (new solution, tabu attribute), or None if infeasible."""
    new = [[st, list(route)] for st, route in solution]
    r = len(new)
    move = rng.choice(["swap_order", "reverse", "exchange", "relocate", "merge"])
    if r < 2:
        move = rng.choice(["swap_order", "reverse"])

    if move in ("swap_order", "reverse"):           # intra-route: swap two customers / 2-opt
        a = rng.integers(r)
        route = new[a][1]
        if len(route) < 2:
            return None
        i, j = sorted(rng.choice(len(route), 2, replace=False))
        attr = frozenset((route[i], route[j]))
        if move == "swap_order":
            route[i], route[j] = route[j], route[i]
        else:
            route[i:j + 1] = route[i:j + 1][::-1]
        changed = [a]
    elif move == "exchange":                        # exchange customers between two routes
        a, b = rng.choice(r, 2, replace=False)
        i, j = rng.integers(len(new[a][1])), rng.integers(len(new[b][1]))
        ci, cj = new[a][1][i], new[b][1][j]
        new[a][1][i], new[b][1][j] = cj, ci
        attr = frozenset((ci, cj))
        changed = [a, b]
    elif move == "relocate":                        # move a customer to another route
        a, b = rng.choice(r, 2, replace=False)
        c = new[a][1].pop(rng.integers(len(new[a][1])))
        new[b][1].insert(rng.integers(len(new[b][1]) + 1), c)
        attr = frozenset((c,))
        changed = [b]
    else:                                           # merge two drones of the same station
        a, b = rng.choice(r, 2, replace=False)
        if new[a][0] != new[b][0]:
            return None
        new[a][1] = sorted(new[a][1] + new[b][1], key=lambda i: -p.demand[i])  # heaviest first
        new[b][1] = []
        attr = None
        changed = [a]

    for k in changed:                               # capacity/battery check of changed routes
        st, route = new[k]
        if not all(p.reachable[i, st] for i in route) or not route_feasible(p, st, route):
            return None
    new = [[st, route] for st, route in new if route]   # empty drones are removed
    return new, attr


def tabu_search(p, start, archive, counter, rng, iterations=25, n_neighbors=30):
    tenure = max(3, min(10, p.n // 4))              # how long a move stays tabu
    tabu = {}                                       # attribute -> iteration when it expires
    current, f_current = start, objectives(p, start)
    for it in range(iterations):
        candidates = []
        for _ in range(n_neighbors):
            if counter[0] <= 0:
                break
            nb = neighbor(p, current, rng)
            if nb is None:
                continue
            counter[0] -= 1
            candidates.append((objectives(p, nb[0]), nb[0], nb[1]))
        if not candidates:
            break
        # non-tabu moves + aspiration (accept a tabu move if it is good enough for the archive)
        allowed = [c for c in candidates
                   if c[2] is None or tabu.get(c[2], -1) < it or
                   not any(np.all(g <= c[0]) for g, _ in archive)]
        if not allowed:
            continue
        # selection by PARETO DOMINANCE instead of a single cost
        F = [c[0] for c in allowed]
        non_dominated = [allowed[i] for i in fronts(F)[0]]
        for f, s, _ in non_dominated:
            add_to_archive(archive, f, s)
        better = [c for c in non_dominated if dominates(c[0], f_current)]
        options = better or non_dominated
        f_current, current, attr = options[rng.integers(len(options))]
        if attr is not None:
            tabu[attr] = it + tenure
    return current


# ------------------------------------------------------------------ DE / DE-TS
def run(p, budget=20000, hybrid=True, seed=0, NP=20, F=0.5, CR=0.9,
        ls_every=5, ls_count=3, ls_ratio=0.5):
    """hybrid=True -> DE-TS (ours), hybrid=False -> DSDE (benchmark).
    Returns the Pareto archive as [(f, solution), ...]."""
    rng = np.random.default_rng(seed)
    dim = p.m + 2 * p.n
    counter = [budget]                               # remaining evaluations
    tabu_used = 0
    archive = []

    X = rng.random((NP, dim))
    solutions = [decode(p, x) for x in X]
    Fo = np.array([objectives(p, s) for s in solutions])
    counter[0] -= NP
    for f, s in zip(Fo, solutions):
        add_to_archive(archive, f, s)

    gen = 0
    while counter[0] > 0:
        gen += 1
        newX, newF, newS = list(X), list(Fo), list(solutions)
        for i in range(NP):
            if counter[0] <= 0:
                break
            # mutation: 70% from the 5 nearest neighbors in decision space (niche), else whole population
            if rng.random() < 0.7:
                pool = np.argsort(np.linalg.norm(X - X[i], axis=1))[1:6]
            else:
                pool = [j for j in range(NP) if j != i]
            r1, r2, r3 = rng.choice(pool, 3, replace=False)
            v = np.clip(X[r1] + F * (X[r2] - X[r3]), 0, 0.999999)
            mask = rng.random(dim) < CR                          # crossover
            mask[rng.integers(dim)] = True
            u = np.where(mask, v, X[i])

            s = decode(p, u)
            f = objectives(p, s)
            counter[0] -= 1
            add_to_archive(archive, f, s)
            if dominates(f, Fo[i]):
                newX[i], newF[i], newS[i] = u, f, s
            elif not dominates(Fo[i], f):
                newX.append(u); newF.append(f); newS.append(s)

        # selection: non-dominated sorting + crowding distance in DECISION SPACE (niching)
        newX, newF = np.array(newX), np.array(newF)
        selected = []
        for front in fronts(newF):
            if len(selected) + len(front) <= NP:
                selected += front
            else:
                d = crowding(newX[front])
                selected += [front[j] for j in np.argsort(-d)[:NP - len(selected)]]
                break
        X, Fo, solutions = newX[selected], newF[selected], [newS[j] for j in selected]

        # HYBRID: every ls_every generations, apply Tabu Search to solutions picked from the archive
        if hybrid and gen % ls_every == 0 and tabu_used <= ls_ratio * (budget - counter[0]):
            Fa = np.array([f for f, _ in archive])
            picks = {int(np.argmin(Fa[:, 0])), int(np.argmin(Fa[:, 1]))}   # the two extremes
            while len(picks) < min(ls_count, len(archive)):
                picks.add(int(rng.integers(len(archive))))
            starts = [archive[k][1] for k in picks]
            for start in starts:
                before = counter[0]
                improved = tabu_search(p, start, archive, counter, rng)
                tabu_used += before - counter[0]
                # replace the worst individual with the improved solution
                worst = fronts(Fo)[-1][0]
                X[worst] = encode(p, improved, rng)
                Fo[worst] = objectives(p, improved)
                solutions[worst] = improved
    return archive


if __name__ == "__main__":
    p = generate_instance(30, seed=7)
    for name, h in [("DSDE", False), ("DE-TS", True)]:
        archive = run(p, budget=10000, hybrid=h)
        print(name, "number of Pareto solutions:", len(archive))
        for f, s in sorted(archive, key=lambda a: a[0][0]):
            print(f"   f1={f[0]:.0f}  f2={f[1]:.1f}  drones={len(s)}")
