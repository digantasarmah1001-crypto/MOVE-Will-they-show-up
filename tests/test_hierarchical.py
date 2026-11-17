import numpy as np
import pandas as pd
import xarray as xr

from thesis.models.hierarchical import codes, online, predict


def test_new_levels_come_after_trained_ones():
    tr, te = codes(pd.Series(["a", "b", "a"]), pd.Series(["b", "c"]))
    assert list(tr) == [0, 1, 0] and list(te) == [1, 2]


def test_predict_known_and_new_patient():
    n = 4000
    post = xr.Dataset({"a": ("s", np.zeros(n)), "b": (("feature", "s"), np.zeros((1, n))),
                       "sigma_p": ("s", np.ones(n)), "z_p": (("lvl", "s"), np.full((1, n), 3.0))})
    p = predict(post, pd.DataFrame({"x": [0.0, 0.0]}), {"p": np.array([0, 1])})
    assert np.allclose(p[:, 0], 1 / (1 + np.exp(-3)))  # known patient keeps their effect
    assert abs(p[:, 1].mean() - 0.5) < 0.02 and p[:, 1].std() > 0.1  # new patient: average over N(0, 1)


def test_online_update_uses_only_earlier_visits():
    p = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])  # two equally likely draws
    out = online(p, y=np.array([1, 0, 1]), person=np.array([7, 7, 8]), day=np.array([1, 2, 1]))
    assert np.isclose(out[0], 0.5) and np.isclose(out[2], 0.5)  # nothing earlier to learn from
    assert np.isclose(out[1], (0.1 * 0.1 + 0.9 * 0.9) / (0.1 + 0.9))  # after a miss on day 1
