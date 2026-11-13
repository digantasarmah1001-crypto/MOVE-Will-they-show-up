from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thesis.data.datasets import hotel_data, noshow_data
from thesis.data.features import split
from thesis.eval.metrics import bootstrap_diff, evaluate
from thesis.models.move import fit_prior

root = Path(__file__).resolve().parent.parent
lgbm = dict(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0)

rows = []
for name, load, split_date in [("noshow", noshow_data, "2016-05-25"), ("hotel", hotel_data, "2016-12-31")]:
    df, X, y, shows, visits, groups, strata, date_col = load()
    tr, te = split(df, date_col, split_date)
    tr, te = tr.index, te.index

    a, b = fit_prior(shows[tr].values, visits[tr].values, y[tr].values)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    dummies = pd.get_dummies(X, drop_first=True, dtype=float)
    preds = {
        "base rate": np.full(len(te), y[tr].mean()),
        "MOVE C": ((shows[te] + 2) / (visits[te] + 3)).values,
        "tuned MOVE C": ((shows[te] + a) / (visits[te] + a + b)).values,
        "logistic regression": lr.fit(dummies.loc[tr], y[tr]).predict_proba(dummies.loc[te])[:, 1],
        "LightGBM": LGBMClassifier(**lgbm).fit(X.loc[tr], y[tr]).predict_proba(X.loc[te])[:, 1],
        "LightGBM + isotonic": CalibratedClassifierCV(LGBMClassifier(**lgbm), method="isotonic", cv=5)
            .fit(X.loc[tr], y[tr]).predict_proba(X.loc[te])[:, 1],
    }

    pd.DataFrame(preds, index=te).to_parquet(root / f"data/processed/preds_{name}.parquet")

    for model, p in preds.items():
        diff = bootstrap_diff(y[te].values, p, preds["MOVE C"], groups[te].values) if model != "MOVE C" else [np.nan] * 3
        rows.append({"dataset": name, "model": model, "stratum": "all", "n": len(te), **evaluate(y[te], p),
                     "vs_MOVE": diff[0], "ci_low": diff[1], "ci_high": diff[2]})
        for s in strata.cat.categories:
            m = (strata[te] == s).values
            rows.append({"dataset": name, "model": model, "stratum": s, "n": m.sum(), **evaluate(y[te][m], p[m])})

out = pd.DataFrame(rows)
out.to_csv(root / "results/tables/baselines.csv", index=False)
print(out[out.stratum == "all"].drop(columns="stratum").round(3).to_string(index=False))
print(out.pivot_table(index=["dataset", "model"], columns="stratum", values="log_loss", sort=False).round(3).to_string())
