import numpy as np
import pandas as pd
from scipy.special import expit, logit

from thesis.models.move import reliability
from thesis.opt.overbooking import best_invites, value

COSTS = {"reward": 1.0, "over_cost": 1.0, "fail_cost": 2.0}


def weighted(history, day, half_life=30):
    # recency-weighted shows and visits from a history of (day, showed)
    w = np.array([0.5 ** ((day - d) / half_life) for d, _ in history])
    return w @ np.array([s for _, s in history], dtype=float), w.sum()


def move_scorer(histories, day):
    return np.array([[reliability([day - d for d, _ in h], [s for _, s in h]) for h in histories]])


def beta_scorer(a, b, half_life=30, n_draws=200, seed=0):
    # the learned score: prior Beta(a, b) updated with each person's weighted history
    rng = np.random.default_rng(seed)
    def score(histories, day):
        shows, visits = np.array([weighted(h, day, half_life) for h in histories]).T
        return rng.beta(a + shows, b + visits - shows, size=(n_draws, len(histories)))
    return score


def invite_capacity(draws, capacity, min_size, **costs):
    return capacity


def invite_fixed_ratio(ratio):
    return lambda draws, capacity, min_size, **costs: int(np.ceil(capacity * ratio))


def invite_optimised(draws, capacity, min_size, **costs):
    return best_invites(draws, capacity, min_size, **costs)[0]


def simulate(scorer, rule, n_users=500, days=90, plans_per_day=10, pool=20, capacity=6, min_size=4,
             mean_show=0.8, sigma=1.0, fair_days=None, true_p=None, reform=0.0, reform_day=30, costs=COSTS, seed=0):
    rng = np.random.default_rng(seed)
    if true_p is None:
        true_p = expit(logit(mean_show) + sigma * rng.standard_normal(n_users))
    n_users = len(true_p)
    reformed = rng.random(n_users) < reform  # these show up with chance 0.3 until reform_day
    histories = [[] for _ in range(n_users)]
    last_invited = np.zeros(n_users)
    plans = []
    for day in range(days):
        for _ in range(plans_per_day):
            ids = rng.choice(n_users, pool, replace=False)
            draws = scorer([histories[i] for i in ids], day)
            order = np.argsort(-draws.mean(0))
            if fair_days:
                due = day - last_invited[ids[order]] >= fair_days
                order = np.concatenate([order[due], order[~due]])
            n = rule(draws[:, order], capacity, min_size, **costs)
            invited = ids[order[:n]]
            came = rng.random(n) < np.where(reformed[invited] & (day < reform_day), 0.3, true_p[invited])
            for i, c in zip(invited, came):
                histories[i].append((day, int(c)))
            last_invited[invited] = day
            k = came.sum()
            plans.append({"day": day, "invited": n, "came": k, "happened": k >= min_size,
                          "overflow": max(k - capacity, 0), "value": float(value(k, capacity, min_size, **costs))})
    return pd.DataFrame(plans), histories, true_p, reformed


def calibrate(mean_show, sigma, n=100_000):
    # simulated people differ like the patients in the no-show model. The learned prior is the Beta
    # with the same mean and variance, which is what enough data would teach the score
    population = expit(logit(mean_show) + sigma * np.random.default_rng(0).standard_normal(n))
    m, v = population.mean(), population.var()
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k, population
