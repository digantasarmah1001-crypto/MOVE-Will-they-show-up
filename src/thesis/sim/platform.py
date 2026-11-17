import numpy as np
import pandas as pd
from scipy.special import expit, logit

from thesis.models.move import reliability
from thesis.opt.overbooking import best_invites


# scorers: take each pool member's history [(day, showed), ...] and return show probabilities (draws, people)
def move_scorer(histories, day):
    return np.array([[reliability([day - d for d, _ in h], [s for _, s in h]) for h in histories]])


def beta_scorer(a, b, half_life=30, n_draws=200, seed=0):
    # learned version of MOVE: prior Beta(a, b) from the data, updated with recency-weighted history
    rng = np.random.default_rng(seed)

    def score(histories, day):
        w = [np.array([0.5 ** ((day - d) / half_life) for d, _ in h]) for h in histories]
        shows = np.array([wi @ np.array([s for _, s in h], dtype=float) if h else 0 for wi, h in zip(w, histories)])
        total = np.array([wi.sum() for wi in w])
        return rng.beta(a + shows, b + total - shows, size=(n_draws, len(histories)))
    return score


# invite rules: take the pool's draws, sorted best first, and return how many to invite
def invite_capacity(draws, capacity, min_size):
    return capacity


def invite_fixed_ratio(ratio):
    return lambda draws, capacity, min_size: int(np.ceil(capacity * ratio))


def invite_optimised(draws, capacity, min_size):
    return best_invites(draws, capacity, min_size)[0]


def simulate(scorer, rule, n_users=500, days=90, plans_per_day=10, pool=20, capacity=6, min_size=4,
             mean_show=0.8, sigma=1.0, fair_days=None, true_p=None, seed=0):
    rng = np.random.default_rng(seed)
    if true_p is None:
        true_p = expit(logit(mean_show) + sigma * rng.standard_normal(n_users))
    n_users = len(true_p)
    histories = [[] for _ in range(n_users)]
    last_invited = np.zeros(n_users)
    plans = []

    for day in range(days):
        for _ in range(plans_per_day):
            ids = rng.choice(n_users, pool, replace=False)
            draws = scorer([histories[i] for i in ids], day)
            order = np.argsort(-draws.mean(0))
            if fair_days:  # anyone not invited for fair_days goes to the front, so low scores can recover
                order = sorted(order, key=lambda j: day - last_invited[ids[j]] < fair_days)
            n = rule(draws[:, order], capacity, min_size)
            invited = ids[order[:n]]
            came = rng.random(n) < true_p[invited]
            for i, c in zip(invited, came):
                histories[i].append((day, int(c)))
            last_invited[invited] = day
            k = came.sum()
            plans.append({"day": day, "invited": n, "came": k, "happened": k >= min_size, "overflow": max(k - capacity, 0),
                          "value": -2.0 if k < min_size else min(k, capacity) - max(k - capacity, 0)})  # same costs as the optimiser
    return pd.DataFrame(plans), histories, true_p
