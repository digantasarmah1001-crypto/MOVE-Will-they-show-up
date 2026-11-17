import numpy as np

from thesis.opt.matching import match_greedy, match_ilp


def check(out, p, interest, capacity, min_size):
    for i, j in enumerate(out):
        assert j == -1 or interest[i, j]  # only plans they like, at most one each
    for j in set(out) - {-1}:
        assert min_size <= p[out == j].sum() <= capacity


def test_both_respect_the_rules():
    rng = np.random.default_rng(0)
    p, interest = rng.uniform(0.4, 1, 30), (rng.random((30, 3)) < 0.5).astype(int)
    check(match_ilp(p, interest, 6, 4, 8), p, interest, 6, 4.5)
    check(match_greedy(p, interest, 6, 4, 8), p, interest, 6, 0)  # greedy has no minimum


def test_ilp_opens_a_plan_greedy_misses():
    # greedy fills plan 0 with the two people who could also have saved plan 1
    p = np.full(6, 0.9)
    interest = np.array([[1, 1], [1, 1], [1, 0], [1, 0], [0, 1], [0, 1]])
    assert len(set(match_ilp(p, interest, 2, 1, 4)) - {-1}) == 2
