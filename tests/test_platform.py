import numpy as np

from thesis.sim.platform import beta_scorer, invite_capacity, move_scorer, simulate


def test_reliable_users_always_meet():
    plans = simulate(move_scorer, invite_capacity, true_p=np.ones(50), days=3)[0]
    assert plans.happened.all() and (plans.overflow == 0).all() and (plans.came == 6).all()


def test_same_seed_same_result():
    a = simulate(beta_scorer(6, 2), invite_capacity, n_users=100, days=5)[0]
    b = simulate(beta_scorer(6, 2), invite_capacity, n_users=100, days=5)[0]
    assert a.equals(b)
