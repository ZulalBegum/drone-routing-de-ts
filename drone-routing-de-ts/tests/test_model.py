"""Run with:  python -m pytest -q"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hybrid import Problem, decode, encode, evaluate, run, dimension, drone_trip, truck_routes, delivery_loads, \
    BATTERY, DRONE_CAP

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def test_matches_article_reference_solution():
    """A non-dominated solution of the authors' ALNS archive for Ca1-2,3,15 has objectives (25.94, 1.6115);
    the article's Table 6 reports [25.9, 1.6] for this instance. Our model must give the same numbers."""
    p = Problem(os.path.join(DATA, "Ca1-2,3,15.csv"))
    sol = [[1, [6, 3, 8]], [1, [4, 0, 11, 13, 9, 7]], [1, [2, 1]], [1, [12, 14, 10, 5]]]
    f, cv = evaluate(p, sol)
    assert np.allclose(f, [25.94, 1.6115], atol=1e-3)
    assert cv == 0


def test_decoder_respects_hard_constraints():
    p = Problem(os.path.join(DATA, "Ca2-3,5,30.csv"))
    rng = np.random.default_rng(0)
    for _ in range(50):
        sol = decode(p, rng.random(dimension(p)))
        assert sorted(c for _, r in sol for c in r) == list(range(p.n))
        ready = truck_routes(p, delivery_loads(p, sol))[2]
        for s, r in sol:
            e, late, load, wait = drone_trip(p, s, r, ready[s])
            assert e <= BATTERY + 1e-9 and load <= DRONE_CAP + 1e-9


def test_encode_roundtrip_is_valid():
    p = Problem(os.path.join(DATA, "Ca1-2,3,30.csv"))
    rng = np.random.default_rng(1)
    sol = decode(p, rng.random(dimension(p)))
    x = encode(p, sol, rng)
    assert x.shape == (dimension(p),) and np.all((x >= 0) & (x <= 1))
    assert sorted(c for _, r in decode(p, x) for c in r) == list(range(p.n))


def test_algorithms_return_feasible_nondominated_archives():
    p = Problem(os.path.join(DATA, "Ca1-2,3,15.csv"))
    for hybrid in (False, True):
        archive = run(p, budget=600, hybrid=hybrid, seed=0)
        assert archive
        for f, sol in archive:
            g, cv = evaluate(p, sol)
            assert cv == 0 and np.allclose(f, g)
        F = np.array([f for f, _ in archive])
        for i in range(len(F)):
            assert not any(np.all(F[j] <= F[i]) and np.any(F[j] < F[i]) for j in range(len(F)))
