import numpy as np
from scipy.stats import poisson_binom

from thesis.opt.overbooking import best_invites, plan_value


def test_matches_simulation():
    p = np.array([0.9, 0.6, 0.8, 0.5, 0.95])
    shows = (np.random.default_rng(0).random((200_000, 5)) < p).sum(1)
    assert np.allclose(np.bincount(shows, minlength=6) / 200_000, poisson_binom.pmf(np.arange(6), p), atol=0.005)


def test_sure_people_fill_capacity_exactly():
    assert best_invites(np.ones(10), capacity=4, min_size=2)[0] == 4


def test_flaky_people_need_more_invites():
    n_reliable = best_invites(np.full(15, 0.9), capacity=6, min_size=4)[0]
    n_flaky = best_invites(np.full(15, 0.5), capacity=6, min_size=4)[0]
    assert 6 <= n_reliable < n_flaky


def test_uncertainty_is_averaged():
    # two equally likely worlds: everyone shows or nobody does. A plug-in p of 0.5 gives a different answer
    draws = np.array([[1.0] * 4, [0.0] * 4])
    assert plan_value(draws, 4, 2) == (4 - 2) / 2
    assert plan_value(np.full(4, 0.5), 4, 2) != plan_value(draws, 4, 2)
