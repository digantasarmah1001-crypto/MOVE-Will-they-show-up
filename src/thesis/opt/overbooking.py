import numpy as np
from scipy.stats import poisson_binom


def value(k, capacity, min_size, reward=1.0, over_cost=1.0, fail_cost=2.0):
    # k people came: too few and the plan fails, too many and the extra ones are turned away
    return np.where(k < min_size, -fail_cost, reward * np.minimum(k, capacity) - over_cost * np.maximum(k - capacity, 0))


def plan_value(p, capacity, min_size, **costs):
    # expected value when everyone in p is invited. p has one row per posterior draw
    p = np.atleast_2d(p)
    k = np.arange(p.shape[1] + 1)
    return (value(k, capacity, min_size, **costs) @ poisson_binom.pmf(k[:, None], p)).mean()


def best_invites(p, capacity, min_size, **costs):
    # people are invited in column order
    p = np.atleast_2d(p)
    values = [plan_value(p[:, :n], capacity, min_size, **costs) for n in range(1, p.shape[1] + 1)]
    return int(np.argmax(values)) + 1, values
