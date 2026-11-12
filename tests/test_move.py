# the same 12 checks as tests.js in the prototype
from thesis.models.move import (ATTENDANCE, PRIOR_MEAN, bayesian_average, label, reliability, show_up_score,
                                simple_average)


def history(events):
    days = [d for d, _, _ in events]
    att = [ATTENDANCE[o] for _, o, _ in events]
    ratings = [r for _, o, r in events if o == "showed"]
    return days, att, ratings


def score(events):
    return show_up_score(*history(events))


veteran = [(i * 2, "showed", 4.6) for i in range(40)]
newcomer = [(0, "showed", 5)]
with_no_show = veteran + [(1, "no_show", None)]
with_late = veteran + [(1, "late", None)]
shows = [(i * 5, "showed", 4) for i in range(10)]


def rel(events):
    return reliability(*history(events)[:2])


def test_empty_history():
    assert bayesian_average([]) == PRIOR_MEAN
    assert abs(reliability([], []) - 2 / 3) < 1e-9
    assert score([]) == 65 and label(65) == "Building trust"


def test_one_review_vs_long_history():
    assert simple_average(history(newcomer)[2]) > simple_average(history(veteran)[2])
    assert bayesian_average(history(newcomer)[2]) < bayesian_average(history(veteran)[2])
    assert score(newcomer) < score(veteran)


def test_no_shows_and_late_cancels():
    assert simple_average(history(with_no_show)[2]) == simple_average(history(veteran)[2])
    assert score(with_no_show) < score(veteran)
    assert rel(with_no_show) < rel(with_late) < rel(veteran)


def test_recency_and_labels():
    assert score(shows + [(90, "no_show", None)]) > score(shows + [(0, "no_show", None)])
    assert abs(reliability([0, 30], [1, 0], a=0, b=0) - 1 / 1.5) < 1e-12  # a 30 day old event counts half
    assert [label(s) for s in (75, 74, 50, 49)] == ["Trusted", "Building trust", "Building trust", "Low show-up"]
