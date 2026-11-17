import numpy as np
from scipy.stats import poisson_binom


def plan_value(p, capacity, min_size, reward=1.0, over_cost=1.0, fail_cost=2.0):
    # expected value of inviting everyone in p. p is (draws, people), one row per posterior draw.
    # k people show up: below min_size the plan fails, above capacity the extra people are turned away
    p = np.atleast_2d(p)
    k = np.arange(p.shape[1] + 1)
    pmf = poisson_binom.pmf(k[:, None], p)  # (k, draws)
    value = np.where(k < min_size, -fail_cost, reward * np.minimum(k, capacity) - over_cost * np.maximum(k - capacity, 0))
    return (value @ pmf).mean()


def best_invites(p, capacity, min_size, **costs):
    # people are invited in the order of the columns. Try every n and keep the best one
    p = np.atleast_2d(p)
    values = [plan_value(p[:, :n], capacity, min_size, **costs) for n in range(1, p.shape[1] + 1)]
    return int(np.argmax(values)) + 1, values
