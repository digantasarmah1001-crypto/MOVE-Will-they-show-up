import numpy as np
from ortools.linear_solver import pywraplp

from thesis.opt.overbooking import value


def match_ilp(p, interest, capacity, min_size, max_invites, due=None, fair_bonus=0.1):
    # first open as many plans as possible, then maximise expected attendance
    due = np.zeros(len(p)) if due is None else due
    s = pywraplp.Solver.CreateSolver("SCIP")
    x = {(i, j): s.BoolVar(f"x_{i}_{j}") for i, j in zip(*np.nonzero(interest))}
    y = [s.BoolVar(f"y_{j}") for j in range(interest.shape[1])]
    s.Maximize(100 * sum(y) + sum((p[i] + fair_bonus * due[i]) * v for (i, j), v in x.items()))
    for i in range(len(p)):
        s.Add(sum(v for (u, j), v in x.items() if u == i) <= 1)
    for j, open_ in enumerate(y):
        invited = [(i, v) for (i, k), v in x.items() if k == j]
        s.Add(sum(v for _, v in invited) <= max_invites * open_)
        s.Add(sum(p[i] * v for i, v in invited) >= (min_size + 0.5) * open_)
        s.Add(sum(p[i] * v for i, v in invited) <= capacity)
    s.SetTimeLimit(5000)
    s.Solve()
    out = np.full(len(p), -1)
    for (i, j), v in x.items():
        if v.solution_value() > 0.5:
            out[i] = j
    return out


def match_greedy(p, interest, capacity, min_size, max_invites, due=None, fair_bonus=0.1):
    # each plan in turn takes its best free users until expected attendance reaches capacity
    due = np.zeros(len(p)) if due is None else due
    out = np.full(len(p), -1)
    for j in range(interest.shape[1]):
        expected = 0
        for i in np.argsort(-(p + fair_bonus * due)):
            if interest[i, j] and out[i] == -1 and expected + p[i] <= capacity and (out == j).sum() < max_invites:
                out[i], expected = j, expected + p[i]
    return out


def simulate_matching(scorer, match, n_users=500, days=60, available=120, plans=10, interests=3, capacity=6,
                      min_size=4, max_invites=8, true_p=None, fair_days=None, seed=0):
    rng = np.random.default_rng(seed)
    histories = [[] for _ in range(n_users)]
    last = np.zeros(n_users)
    rows = []
    for day in range(days):
        ids = rng.choice(n_users, available, replace=False)
        interest = np.zeros((available, plans), dtype=int)
        for r in range(available):
            interest[r, rng.choice(plans, interests, replace=False)] = 1
        p = scorer([histories[i] for i in ids], day).mean(0)
        due = (day - last[ids] >= fair_days).astype(float) if fair_days else None
        plan_of = match(p, interest, capacity, min_size, max_invites, due)
        came = rng.random(available) < true_p[ids]
        for r in np.flatnonzero(plan_of >= 0):
            histories[ids[r]].append((day, int(came[r])))
            last[ids[r]] = day
        for j in range(plans):
            k = (came & (plan_of == j)).sum()
            rows.append({"day": day, "invited": (plan_of == j).sum(), "came": k, "happened": k >= min_size,
                         "value": float(value(k, capacity, min_size))})
    return rows
