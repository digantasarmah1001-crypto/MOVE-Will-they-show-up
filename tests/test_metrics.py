import numpy as np
import pandas as pd

from thesis.eval.metrics import evaluate


def test_well_calibrated_predictions_have_slope_one():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 50_000)
    y = (rng.random(50_000) < p).astype(int)
    m = evaluate(pd.Series(y), pd.Series(p))
    assert abs(m["cal_slope"] - 1) < 0.05 and abs(m["cal_intercept"]) < 0.05
