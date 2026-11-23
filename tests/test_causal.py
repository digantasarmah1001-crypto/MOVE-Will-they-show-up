import numpy as np
import pandas as pd

from thesis.models.causal import balance, diff_in_diff, e_value, effects, propensity


def confounded(n=40_000, seed=0):
    # c makes treatment more likely and the outcome more likely, the true effect is -0.10
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n)
    t = (rng.random(n) < 1 / (1 + np.exp(-c))).astype(int)
    y = (rng.random(n) < 0.3 + 0.1 * (c > 0) - 0.1 * t).astype(int)
    return pd.DataFrame({"c": c, "c_pos": (c > 0).astype(float)}), t, y


def test_estimators_remove_confounding():
    X, t, y = confounded()
    naive = y[t == 1].mean() - y[t == 0].mean()
    est = effects(X, t, y)
    assert abs(naive + 0.10) > 0.03
    for name in ["weighting", "regression", "doubly robust"]:
        assert abs(est[name] + 0.10) < 0.02


def test_weighting_balances_the_confounder():
    X, t, y = confounded()
    ps = propensity(X, t)
    w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
    assert abs(balance(X, t)["c"]) > 0.3 and abs(balance(X, t, w)["c"]) < 0.05


def test_e_value():
    assert np.isclose(e_value(2.0), 2 + np.sqrt(2)) and np.isclose(e_value(0.5), e_value(2.0))


def test_diff_in_diff_ignores_shared_shocks():
    # the outage raises everyone's risk by 0.02, and the exposed group's by a further 0.05
    rng = np.random.default_rng(1)
    n = 60_000
    exposed, after = rng.integers(0, 2, n), rng.integers(0, 2, n)
    y = (rng.random(n) < 0.2 + 0.1 * exposed + 0.02 * after + 0.05 * exposed * after).astype(int)
    assert abs(diff_in_diff(pd.DataFrame(index=range(n)), y, exposed, after) - 0.05) < 0.015
