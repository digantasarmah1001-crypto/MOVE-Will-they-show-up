from thesis.sim.gaming import personas, recovery, whitewash


def test_learned_score_sees_through_the_rating_farmer():
    t = personas(6.3, 1.7).set_index("persona")
    assert t.loc["Nico (rating farmer)", "A rating mean"] == 5
    assert t.loc["Nico (rating farmer)", "learned P(show)"] < t.loc["Marta (reliable)", "learned P(show)"]


def test_a_user_who_never_shows_gains_from_a_new_account():
    w = whitewash([[(0, 0), (5, 0), (9, 0)]], [0.1], day=10, a=6.3, b=1.7)
    assert (w.gain > 0).all()


def test_recovery_takes_at_least_one_plan():
    assert all(n >= 1 for n in recovery(6.3, 1.7, target=0.8).values())
