from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thesis.data.features import add_history, split
from thesis.eval.metrics import bootstrap_diff, evaluate
from thesis.models.move import fit_prior

root = Path(__file__).resolve().parent.parent
lgbm = dict(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0)


def noshow_data():
    df = add_history(pd.read_parquet(root / "data/processed/noshow.parquet"))
    X = df[["age", "scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received",
            "prior_visits", "prior_no_shows", "w_visits", "w_shows"]].assign(
        log_lead=np.log1p(df.lead_days), female=(df.gender == "F").astype(int),
        log_days_since_last=np.log1p(df.days_since_last.fillna(0)),
        weekday=df.appointment_day.dt.day_name().astype("category"), neighbourhood=df.neighbourhood.astype("category"))
    strata = pd.cut(df.prior_visits, [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"])
    # MOVE: weighted shows and visits, no late cancels in this data
    return df, X, 1 - df.no_show, df.w_shows, df.w_visits, df.patient_id, strata, "appointment_day"


def hotel_data():
    df = pd.read_parquet(root / "data/processed/hotel.parquet")
    cats = ["hotel", "arrival_date_month", "meal", "country", "market_segment", "distribution_channel",
            "reserved_room_type", "deposit_type", "agent", "company", "customer_type"]
    X = df[["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies", "group_size",
            "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "adr"]].assign(
        log_lead=np.log1p(df.lead_time), **{c: df[c].astype("category") for c in cats})
    prev = df.previous_cancellations + df.previous_bookings_not_canceled
    strata = pd.cut(prev, [-1, 0, 2, np.inf], labels=["0", "1-2", "3+"])
    df["date"] = df.arrival_date
    return df, X, 1 - df.is_canceled, df.previous_bookings_not_canceled, prev, df.index, strata, "date"


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
