import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


def calibration(y, p):
    # slope 1 and intercept 0 mean well calibrated
    x = logit(np.clip(p, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    fit = LogisticRegression(C=np.inf).fit(x, y)
    return fit.coef_[0, 0], fit.intercept_[0]


def evaluate(y, p):
    slope, intercept = calibration(y, p) if np.ptp(p) > 0 else (np.nan, np.nan)
    return {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p),
            "cal_slope": slope, "cal_intercept": intercept, "mean_p": np.mean(p), "rate": np.mean(y)}


def row_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def bootstrap_diff(y, p1, p2, groups, n=1000, seed=0):
    # log loss of p1 minus p2 with a 95% interval, resampling whole people
    per = pd.DataFrame({"d": row_loss(y, p1) - row_loss(y, p2), "g": groups}).groupby("g").d.agg(["sum", "size"])
    idx = np.random.default_rng(seed).integers(0, len(per), (n, len(per)))
    boot = per["sum"].values[idx].sum(1) / per["size"].values[idx].sum(1)
    return per["sum"].sum() / per["size"].sum(), *np.percentile(boot, [2.5, 97.5])
