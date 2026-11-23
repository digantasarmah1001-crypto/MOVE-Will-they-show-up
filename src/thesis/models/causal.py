import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def logistic(X, y):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X, y)


def propensity(X, t):
    return logistic(X, t).predict_proba(X)[:, 1]


def effects(X, t, y, ps=None):
    # effect of the treatment t on the risk of y, as a risk difference, three ways
    ps = propensity(X, t) if ps is None else ps
    outcome = logistic(X.assign(t=t), y)
    m1 = outcome.predict_proba(X.assign(t=1))[:, 1]
    m0 = outcome.predict_proba(X.assign(t=0))[:, 1]
    w1, w0 = t / ps, (1 - t) / (1 - ps)
    return {"weighting": (w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum(),
            "regression": (m1 - m0).mean(),
            "doubly robust": (m1 - m0 + w1 * (y - m1) - w0 * (y - m0)).mean(),
            "risk with": m1.mean(), "risk without": m0.mean()}


def balance(X, t, w=None):
    # standardised mean difference per column, with optional weights. Below 0.1 counts as balanced
    w = np.ones(len(t)) if w is None else w
    treated = np.average(X[t == 1], axis=0, weights=w[t == 1])
    control = np.average(X[t == 0], axis=0, weights=w[t == 0])
    return pd.Series((treated - control) / X.std().replace(0, 1).values, index=X.columns)


def e_value(rr):
    # how strong an unmeasured confounder would need to be to explain the effect away
    rr = 1 / rr if rr < 1 else rr
    return rr + np.sqrt(rr * (rr - 1))


def diff_in_diff(X, y, exposed, after):
    # the exposed group's extra change when `after` is on, adjusted for X (a linear probability model)
    design = X.assign(exposed=exposed, after=after, both=exposed * after)
    return LinearRegression().fit(design, y).coef_[-1]
