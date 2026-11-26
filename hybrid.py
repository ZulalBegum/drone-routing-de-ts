# Two-echelon vehicle-drone routing with load-dependent drones and time windows (2E-VRPLDTW)
# Problem from: Yu et al. (2027), Transportation Research Part E 217, 105195
#
# Method:     DSDE + Pareto Tabu Search hybrid (DE-TS)            -> run(p, hybrid=True)
# Benchmark:  DSDE (our re-implementation of the article's DSDE)  -> run(p, hybrid=False)
#
# 1st echelon: trucks carry delivery parcels from a depot to satellites.
# 2nd echelon: drones leave a satellite, serve customers (delivery or pickup) inside their
#              time windows, and return to the satellite nearest to their last customer.
#
# f1 = dispatch cost    = 12.5 * trucks + 3.36 * drone trips
# f2 = operational cost = 0.1286 * truck km + 0.001 * drone energy (Wh)
# Hard constraints (always kept): drone payload <= 10 kg, drone energy <= 550 Wh
# Soft constraint (CV, must be 0 at the end): time windows (total lateness in minutes)

import csv
from itertools import permutations

import numpy as np

TRUCK_COST, DRONE_COST = 12.5, 3.36
TRUCK_KM_COST, ENERGY_COST = 0.1286, 0.001
TRUCK_CAP, DRONE_CAP = 100.0, 10.0       # kg
DRONE_WEIGHT, BATTERY = 5.0, 550.0       # kg, Wh
TRUCK_SPEED, DRONE_SPEED = 30.0, 10.0    # km/h
POWER_C = 9.81 ** 3 / np.sqrt(2 * 1.225 * np.pi * 10 ** 2 * 4)   # power constant (as in the authors' code)
K_SAT = 2                                # a customer is launched from one of its 2 nearest OPEN satellites


# ------------------------------------------------------------------ PROBLEM
class Problem:
    def __init__(self, path):
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        dep = [r for r in rows if r["type"] == "depot"]
        sat = [r for r in rows if r["type"] == "sat"]
        cus = [r for r in rows if r["type"] == "cust"]
        xy = lambda rs: np.array([[float(r["latitude"]), float(r["longitude"])] for r in rs]) / 10  # km
        self.name = path.replace("\\", "/").split("/")[-1].replace(".csv", "")
        self.depots, self.sats, self.customers = xy(dep), xy(sat), xy(cus)
        self.demand = np.array([float(r["demand"]) for r in cus]) / 10                      # kg
        self.pickup = np.array([r["customer type"] == "pickup" for r in cus])
        self.e = np.array([float(r["start TW"]) for r in cus])                              # minutes
        self.l = np.array([float(r["end TW"]) for r in cus])
        self.service_sat = np.array([float(r["service time"]) for r in sat])
        self.service = np.array([float(r["service time"]) for r in cus])
        self.n, self.S, self.P = len(cus), len(sat), len(dep)

        dist = lambda A, B: np.linalg.norm(A[:, None] - B[None, :], axis=2)
        self.d_cc = dist(self.customers, self.customers)
        self.d_cs = dist(self.customers, self.sats)
        self.d_ds = dist(self.depots, self.sats)
        self.d_ss = dist(self.sats, self.sats)
        self.sat_order = np.argsort(self.d_cs, axis=1)          # satellites from nearest to farthest
        self.return_sat = self.sat_order[:, 0]                  # drone returns to nearest satellite
        self.truck_cache = {}


def power(load):
    """Drone power (W) with a given payload (kg)."""
    return (DRONE_WEIGHT + load) ** 1.5 * POWER_C


# ------------------------------------------------------------------ 1st ECHELON (trucks)
def truck_routes(p, loads):
    """Delivery load per satellite -> (number of trucks, km, satellite ready times, routes).
    Fewest trucks first, then shortest distance (same rule as the authors' code)."""
    used = tuple(s for s in range(p.S) if loads[s] > 1e-9)
    key = (used, round(sum(loads), 6) <= TRUCK_CAP)
    if key in p.truck_cache:
        return p.truck_cache[key]
    groups = [list(used)] if key[1] else first_fit(used, loads)
    trucks, km, routes = 0, 0.0, []
    for g in groups:
        best = None
        for d in range(p.P):
            for order in permutations(g):
                km_r = p.d_ds[d, order[0]] + sum(p.d_ss[a, b] for a, b in zip(order[:-1], order[1:])) \
                    + p.d_ds[d, order[-1]]
                if best is None or km_r < best[0]:
                    best = (km_r, d, order)
        trucks += 1
        km += best[0]
        routes.append((best[1], list(best[2])))
    ready = np.zeros(p.S)                         # time a satellite can launch drones (minutes)
    for d, order in routes:
        t = 60 * p.d_ds[d, order[0]] / TRUCK_SPEED
        for i, s in enumerate(order):
            if i > 0:
                t += p.service_sat[order[i - 1]] + 60 * p.d_ss[order[i - 1], s] / TRUCK_SPEED
            ready[s] = t + p.service_sat[s]
    p.truck_cache[key] = (trucks, km, ready, routes)
    return p.truck_cache[key]


def first_fit(used, loads):
    groups, room = [], []
    for s in sorted(used, key=lambda s: -loads[s]):
        for g in range(len(groups)):
            if loads[s] <= room[g]:
                groups[g].append(s); room[g] -= loads[s]; break
        else:
            groups.append([s]); room.append(TRUCK_CAP - loads[s])
    return groups


# ------------------------------------------------------------------ 2nd ECHELON (drones)
def drone_trip(p, sat, route, ready):
    """Simulates one drone trip. Returns (energy Wh, lateness min, max payload kg, max wait min).
    Take-off is delayed so the drone does not hover before the first customer."""
    load = sum(p.demand[c] for c in route if not p.pickup[c])      # delivery parcels loaded at start
    max_load, max_wait = load, 0.0
    c = route[0]
    fly = 60 * p.d_cs[c, sat] / DRONE_SPEED
    t = max(ready, p.e[c] - fly) + fly
    energy = power(load) * fly / 60
    late = max(0.0, t - p.l[c])
    t = max(t, p.e[c]) + p.service[c]
    load += p.demand[c] if p.pickup[c] else -p.demand[c]
    max_load = max(max_load, load)
    for a, b in zip(route[:-1], route[1:]):
        fly = 60 * p.d_cc[a, b] / DRONE_SPEED
        t += fly
        wait = max(0.0, p.e[b] - t)                                 # hover until time window opens
        energy += power(load) * (fly + wait) / 60
        late += max(0.0, t - p.l[b])
        max_wait = max(max_wait, wait)
        t = max(t, p.e[b]) + p.service[b]
        load += p.demand[b] if p.pickup[b] else -p.demand[b]
        max_load = max(max_load, load)
    last = route[-1]
    energy += power(load) * (60 * p.d_cs[last, p.return_sat[last]] / DRONE_SPEED) / 60
    return energy, late, max_load, max_wait


def trip_ok(p, sat, route, ready):
    e, late, load, wait = drone_trip(p, sat, route, ready)
    return e <= BATTERY + 1e-9 and load <= DRONE_CAP + 1e-9


def delivery_loads(p, sol):
    loads = np.zeros(p.S)
    for sat, route in sol:
        loads[sat] += sum(p.demand[c] for c in route if not p.pickup[c])
    return loads


def evaluate(p, sol):
    """sol = [[launch satellite, [customers]], ...]  ->  (objectives [f1, f2], CV = total lateness)"""
    trucks, km, ready, _ = truck_routes(p, delivery_loads(p, sol))
    energy, late = 0.0, 0.0
    for sat, route in sol:
        e, lt, _, _ = drone_trip(p, sat, route, ready[sat])
        energy += e
        late += lt
    f1 = TRUCK_COST * trucks + DRONE_COST * len(sol)
    f2 = TRUCK_KM_COST * km + ENERGY_COST * energy
    return np.array([f1, f2]), late


def details(p, sol):
    """Everything the user interface needs to draw and list a solution."""
    trucks, km, ready, troutes = truck_routes(p, delivery_loads(p, sol))
    drones = []
    for sat, route in sol:
        e, late, load, _ = drone_trip(p, sat, route, ready[sat])
        drones.append(dict(sat=sat, route=route, end_sat=int(p.return_sat[route[-1]]),
                           energy=e, lateness=late, max_load=load))
    return dict(truck_routes=troutes, truck_km=km, drones=drones)


# ------------------------------------------------------------------ DECODING (random keys)
# x = [satellite open keys (S) | assignment keys (n) | priority keys (n) | insertion bias gene (1)]
SHIFT = 60.0       # a priority key can move a customer up to +-30 min in the insertion order
MAX_WAIT = 200.0   # longest hover (min) accepted when inserting a customer into a trip
MAX_BIAS = 400.0   # Wh: how much extra energy we accept to avoid using one more drone


def dimension(p):
    return p.S + 2 * p.n + 1


def decode(p, x):
    """Cheapest feasible insertion: customers (in priority order) are inserted into the trip of their
    satellite where the extra energy is smallest; a new drone is used only if that is cheaper
    (the bias gene lets DE trade drones (f1) against energy (f2))."""
    S, n = p.S, p.n
    is_open = x[:S] > 0.5
    if not is_open.any():
        is_open[int(np.argmax(x[:S]))] = True
    assign = np.zeros(n, int)
    for c in range(n):
        cands = [s for s in p.sat_order[c] if is_open[s]][:K_SAT]     # nearest open satellites
        assign[c] = cands[min(int(x[S + c] * len(cands)), len(cands) - 1)]
    bias = x[S + 2 * n] * MAX_BIAS
    loads = np.zeros(p.S)
    for c in range(n):
        if not p.pickup[c]:
            loads[assign[c]] += p.demand[c]
    ready = truck_routes(p, loads)[2]
    sol = []
    for s in range(S):
        custs = [c for c in range(n) if assign[c] == s]
        custs.sort(key=lambda c: p.e[c] + (x[S + n + c] - 0.5) * SHIFT)
        trips, info = [], []                    # info[k] = (energy, lateness) of trips[k]
        for c in custs:
            alone = drone_trip(p, s, [c], ready[s])
            best = None
            for k, r in enumerate(trips):
                pos0 = int(np.searchsorted([p.e[v] for v in r], p.e[c]))
                for pos in {max(0, pos0 - 1), pos0, min(len(r), pos0 + 1)}:
                    trial = r[:pos] + [c] + r[pos:]
                    e, late, load, wait = drone_trip(p, s, trial, ready[s])
                    if e > BATTERY or load > DRONE_CAP or wait > MAX_WAIT or late > info[k][1] + alone[1] + 1e-9:
                        continue
                    delta = e - info[k][0]
                    if best is None or delta < best[0]:
                        best = (delta, k, trial, e, late)
            if best is not None and best[0] <= alone[0] + bias:
                trips[best[1]], info[best[1]] = best[2], (best[3], best[4])
            else:
                trips.append([c])
                info.append((alone[0], alone[1]))
        sol += [[s, r] for r in trips]
    return sol


def encode(p, sol, rng):
    """Solution -> random-key vector (to put a Tabu-improved solution back into the DE population)."""
    S, n = p.S, p.n
    x = rng.random(dimension(p))
    used = {s for s, _ in sol}
    x[:S] = [rng.uniform(0.6, 0.9) if s in used else rng.uniform(0.1, 0.4) for s in range(S)]
    for sat, route in sol:
        for c in route:
            cands = [s for s in p.sat_order[c] if s in used][:K_SAT]
            if sat in cands:
                x[S + c] = (cands.index(sat) + 0.5) / len(cands)
    for s in range(S):
        seq = [c for sat, route in sol if sat == s for c in route]
        v_prev = -np.inf
        for c in seq:                                  # make e_c + shift increase along the routes
            v = max(p.e[c], v_prev + 0.01)
            x[S + n + c] = np.clip(0.5 + (v - p.e[c]) / SHIFT, 0, 1)
            v_prev = p.e[c] + (x[S + n + c] - 0.5) * SHIFT
    x[S + 2 * n] = rng.uniform(0.3, 1.0)
    return x


# ------------------------------------------------------------------ PARETO TOOLS
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def cdp(fa, ca, fb, cb):
    """Constrained dominance: smaller violation wins, equal violation -> Pareto dominance."""
    if abs(ca - cb) > 1e-9:
        return ca < cb
    return dominates(fa, fb)


def add_to_archive(archive, f, sol, cv):
    """Archive of FEASIBLE non-dominated solutions: [(f, sol), ...]"""
    if cv > 1e-9:
        return False
    for g, _ in archive:
        if np.all(g <= f):
            return False
    archive[:] = [(g, s) for g, s in archive if not dominates(f, g)] + [(f, sol)]
    return True


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
def neighbor(p, sol, rng):
    """Random small change. Returns (new solution, tabu attribute) or None if payload/battery fail."""
    new = [[s, list(r)] for s, r in sol]
    k = len(new)
    move = rng.choice(["swap_order", "relocate", "exchange", "merge", "new_trip", "move_trip"])
    if k < 2 and move in ("relocate", "exchange", "merge"):
        move = "new_trip"
    if move == "swap_order":                       # swap two customers inside one trip
        a = rng.integers(k)
        r = new[a][1]
        if len(r) < 2:
            return None
        i, j = rng.choice(len(r), 2, replace=False)
        attr = frozenset((r[i], r[j]))
        r[i], r[j] = r[j], r[i]
        changed = [a]
    elif move == "relocate":                       # move a customer to another trip (best position)
        a, b = rng.choice(k, 2, replace=False)
        c = new[a][1].pop(rng.integers(len(new[a][1])))
        rb = new[b][1]
        pos = int(np.searchsorted([p.e[x] for x in rb], p.e[c]))   # time-window order
        rb.insert(pos, c)
        attr = frozenset((c,))
        changed = [b]
    elif move == "exchange":                       # exchange two customers between trips
        a, b = rng.choice(k, 2, replace=False)
        i, j = rng.integers(len(new[a][1])), rng.integers(len(new[b][1]))
        ci, cj = new[a][1][i], new[b][1][j]
        new[a][1][i], new[b][1][j] = cj, ci
        attr = frozenset((ci, cj))
        changed = [a, b]
    elif move == "merge":                          # one drone does two trips' customers
        a, b = rng.choice(k, 2, replace=False)
        new[a][1] = sorted(new[a][1] + new[b][1], key=lambda c: p.e[c])
        new[b][1] = []
        attr = None
        changed = [a]
    elif move == "new_trip":                       # give a customer its own drone (one of its 2 nearest sats)
        a = rng.integers(k)
        c = new[a][1].pop(rng.integers(len(new[a][1])))
        new.append([int(p.sat_order[c][rng.integers(K_SAT)]), [c]])
        attr = frozenset((c,))
        changed = [len(new) - 1]
    else:                                          # launch a whole trip from another satellite
        a = rng.integers(k)
        others = [s for s in range(p.S) if s != new[a][0]]
        new[a][0] = int(others[rng.integers(len(others))])
        attr = ("trip", frozenset(new[a][1]))
        changed = [a]
    new = [[s, r] for s, r in new if r]
    ready = truck_routes(p, delivery_loads(p, new))[2]
    for s, r in new:                               # payload + battery are hard constraints
        if not trip_ok(p, s, r, ready[s]):
            return None
    return new, attr


def tabu_search(p, start, archive, counter, rng, iterations=10, n_neighbors=15):
    tenure = max(3, min(10, p.n // 4))
    tabu = {}
    cur = start
    f_cur, cv_cur = evaluate(p, cur)
    for it in range(iterations):
        cands = []
        for _ in range(n_neighbors):
            if counter[0] <= 0:
                break
            nb = neighbor(p, cur, rng)
            counter[0] -= 1
            if nb is None:
                continue
            f, cv = evaluate(p, nb[0])
            cands.append((f, cv, nb[0], nb[1]))
        if not cands:
            break
        # aspiration: a tabu move is allowed if it would enter the archive
        allowed = [c for c in cands if c[3] is None or tabu.get(c[3], -1) < it or
                   (c[1] <= 1e-9 and not any(np.all(g <= c[0]) for g, _ in archive))]
        if not allowed:
            continue
        # PARETO (constrained) dominance instead of a single cost
        nd = [a for a in allowed if not any(cdp(b[0], b[1], a[0], a[1]) for b in allowed)]
        for f, cv, s, _ in nd:
            add_to_archive(archive, f, s, cv)
        better = [a for a in nd if cdp(a[0], a[1], f_cur, cv_cur)]
        options = better or nd
        f_cur, cv_cur, cur, attr = options[rng.integers(len(options))]
        if attr is not None:
            tabu[attr] = it + tenure
    return cur, f_cur, cv_cur


# ------------------------------------------------------------------ DSDE / DE-TS
def run(p, budget=2000, hybrid=True, seed=0, NP=20, F=0.5,
        ls_every=5, ls_count=2, ls_ratio=0.5):
    """hybrid=False -> DSDE (benchmark), hybrid=True -> DE-TS (ours).
    Returns the archive of feasible non-dominated solutions [(f, solution), ...]."""
    rng = np.random.default_rng(seed)
    dim = dimension(p)
    counter = [budget]
    T_max = budget // NP
    archive, tabu_used = [], 0

    def evaluate_all(X):
        sols = [decode(p, x) for x in X]
        res = [evaluate(p, s) for s in sols]
        counter[0] -= len(X)
        for s, (f, cv) in zip(sols, res):
            add_to_archive(archive, f, s, cv)
        return sols, np.array([r[0] for r in res]), np.array([r[1] for r in res])

    X = rng.random((NP, dim))
    sols, Fo, CV = evaluate_all(X)
    eps0 = CV.max()
    t = 0
    while counter[0] >= NP:
        t += 1
        # --- article's dynamic parameters
        ratio = 1 + np.exp(-15 * (t / (0.8 * T_max) - 0.5))
        eta, beta = 1 - 1 / ratio, 0.5 + 0.5 / ratio
        eps = eps0 * (1 - t / T_max) ** 2 if t < 0.8 * T_max else 0.0
        elite = X[np.lexsort((Fo.sum(axis=1), CV))[0]]

        # --- offspring: DE/rand/1 (CR = 1) + elite-guided / elite-attracted hybrid operator
        idx = np.array([rng.choice([j for j in range(NP) if j != i], 2, replace=False) for i in range(NP)])
        V = X + F * (X[idx[:, 0]] - X[idx[:, 1]])
        guided = rng.random(NP) < eta
        V[guided] += eta * (rng.random((guided.sum(), dim)) - 0.5) * (elite - 0.5)
        V[~guided] += beta * np.abs(elite - V[~guided]) * (elite - V[~guided])
        lo, hi = V.min(axis=1, keepdims=True), V.max(axis=1, keepdims=True)
        V = (V - lo) / np.where(hi - lo > 0, hi - lo, 1) * 0.999999       # row normalization to [0, 1)
        newS, newF, newCV = evaluate_all(V)

        # --- environmental selection: decision-space niche constrained dominance (DSDN-CDP)
        X, Fo, CV, sols = select(np.vstack([X, V]), np.vstack([Fo, newF]), np.concatenate([CV, newCV]),
                                 sols + newS, NP, t, T_max, eps)

        # --- HYBRID: Pareto Tabu Search on archive extremes (+ random), Lamarckian re-injection
        if hybrid and t % ls_every == 0 and tabu_used <= ls_ratio * (budget - counter[0]):
            if archive:
                Fa = np.array([f for f, _ in archive])
                picks = {int(np.argmin(Fa[:, 0])), int(np.argmin(Fa[:, 1]))}
                while len(picks) < min(ls_count, len(archive)):
                    picks.add(int(rng.integers(len(archive))))
                starts = [archive[k][1] for k in picks]
            else:
                starts = [sols[int(np.argmin(CV))]]
            for st in starts:
                before = counter[0]
                cur, f, cv = tabu_search(p, st, archive, counter, rng)
                tabu_used += before - counter[0]
                worst = int(np.lexsort((Fo.sum(axis=1), CV))[-1])
                X[worst], Fo[worst], CV[worst], sols[worst] = encode(p, cur, rng), f, cv, cur
    return archive


def select(X, Fo, CV, sols, NP, t, T_max, eps):
    cv = np.where(CV <= eps, 0.0, CV)                       # dynamic constraint tolerance
    sigma = T_max / 6
    NR = np.sqrt(X.shape[1]) / 3 * (1 - np.exp(-(t - T_max / 2) ** 2 / (2 * sigma ** 2)))
    D = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    n = len(X)
    dom = np.array([[i != j and D[i, j] < NR and cdp(Fo[i], cv[i], Fo[j], cv[j]) for j in range(n)]
                    for i in range(n)])
    rank = np.full(n, -1)
    level, remaining = 0, set(range(n))
    while remaining:
        front = [j for j in remaining if not any(dom[i, j] for i in remaining)]
        for j in front:
            rank[j] = level
        remaining -= set(front)
        level += 1
    keep = []
    for r in range(level):
        front = list(np.where(rank == r)[0])
        if len(keep) + len(front) <= NP:
            keep += front
        else:                                               # truncation: drop most crowded (objective space)
            Fn = (Fo - Fo.min(axis=0)) / np.maximum(np.ptp(Fo, axis=0), 1e-12)
            while len(keep) + len(front) > NP:
                d = np.linalg.norm(Fn[front][:, None] - Fn[front][None, :], axis=2)
                np.fill_diagonal(d, np.inf)
                front.pop(int(np.argmin(d.min(axis=1))))
            keep += front
            break
    return X[keep], Fo[keep], CV[keep], [sols[k] for k in keep]


if __name__ == "__main__":
    import sys
    p = Problem(sys.argv[1] if len(sys.argv) > 1 else "data/Ca1-2,3,15.csv")
    for name, h in [("DSDE", False), ("DE-TS", True)]:
        archive = run(p, hybrid=h)
        print(name, p.name, "Pareto solutions:", len(archive))
        for f, s in sorted(archive, key=lambda a: a[0][0]):
            print(f"   f1={f[0]:.1f}  f2={f[1]:.2f}  drones={len(s)}")
