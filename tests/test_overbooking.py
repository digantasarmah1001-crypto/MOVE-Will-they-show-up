import numpy as np
from scipy.stats import poisson_binom

from thesis.opt.overbooking import best_extra


def test_matches_simulation():
    p = np.array([0.9, 0.6, 0.8, 0.5, 0.95])
    shows = (np.random.default_rng(0).random((200_000, 5)) < p).sum(1)
    assert np.allclose(np.bincount(shows, minlength=6) / 200_000, poisson_binom.pmf(np.arange(6), p), atol=0.005)


def test_clinic_overbooking():
    assert best_extra(np.ones(10), 0.8, slots=10) == 0
    assert best_extra(np.full(10, 0.7), 0.7, slots=10) > 0
    cheap = best_extra(np.full(10, 0.7), 0.7, slots=10, over_cost=0.5)
    dear = best_extra(np.full(10, 0.7), 0.7, slots=10, over_cost=2.0)
    assert cheap > dear
