import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from thesis.data.datasets import hotel_data, noshow_data
from thesis.data.features import split
from thesis.eval.metrics import bootstrap_diff, evaluate
from thesis.models.hierarchical import codes, fit, online, predict, thin

root = Path(__file__).resolve().parent.parent


def noshow_design(df):
    # no past-behaviour features: the patient effect carries history (both together would be biased)
    num = pd.DataFrame({"log_lead": np.log1p(df.lead_days), "age": df.age})
    binary = df[["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]].assign(
        female=(df.gender == "F").astype(int))
    weekday = pd.get_dummies(df.appointment_day.dt.day_name(), dtype=int).drop(columns="Monday")
    return num, pd.concat([binary, weekday], axis=1), ["patient_id", "neighbourhood"]


def hotel_design(df):
    num = pd.DataFrame({"log_lead": np.log1p(df.lead_time), "group_size": df.group_size, "adr": df.adr,
                        "weekend_nights": df.stays_in_weekend_nights, "week_nights": df.stays_in_week_nights,
                        "log_prev_cancellations": np.log1p(df.previous_cancellations),
                        "log_prev_kept": np.log1p(df.previous_bookings_not_canceled)})
    binary = pd.concat([df[["is_repeated_guest"]], (df.hotel == "Resort Hotel").astype(int).rename("resort"),
                        pd.get_dummies(df.deposit_type, dtype=int).drop(columns="No Deposit"),
                        pd.get_dummies(df.customer_type, dtype=int).drop(columns="Transient"),
                        # only 8 segments, too few to estimate a spread, so a normal feature instead of a group
                        pd.get_dummies(df.market_segment, dtype=int).drop(columns="Online TA")], axis=1)
    return num, binary, ["agent", "country", "arrival_date_month"]


datasets = {"noshow": (noshow_data, noshow_design, "2016-05-25", "LightGBM + isotonic", {"draws": 2000}),
            "hotel": (hotel_data, hotel_design, "2016-12-31", "LightGBM", {"draws": 2000, "target_accept": 0.95, "tune": 2000})}
tables = root / "results/tables"

for name in sys.argv[1:] or datasets:
    load, design, split_date, best, sampler = datasets[name]
    rows = []
    df, _, y_show, _, _, people, strata, date_col = load()
    tr, te = (d.index for d in split(df, date_col, split_date))
    num, binary, group_cols = design(df)
    # numeric features scaled with training mean and sd, so effects are "per 1 sd"
    X = pd.concat([(num - num.loc[tr].mean()) / num.loc[tr].std(), binary], axis=1)
    g_tr, g_te = {}, {}
    for g in group_cols:
        g_tr[g], g_te[g] = codes(df.loc[tr, g], df.loc[te, g])

    # a saved model is reused, delete results/models/hier_<name>.nc to refit
    path = root / f"results/models/hier_{name}.nc"
    if path.exists():
        post = xr.open_dataset(path)
    else:
        idata = fit(X.loc[tr], 1 - y_show[tr].values, g_tr, **sampler)
        summ = az.summary(idata, var_names=["a", "b"] + [f"sigma_{g}" for g in group_cols], ci_prob=0.9)
        summ.to_csv(tables / f"hier_effects_{name}.csv")
        z = az.summary(idata, var_names=[f"z_{g}" for g in group_cols], kind="diagnostics")
        pd.DataFrame([{"divergences": int(idata.sample_stats.diverging.sum()),
                       "max_r_hat": max(summ.r_hat.max(), z.r_hat.max()),
                       "min_ess_bulk": min(summ.ess_bulk.min(), z.ess_bulk.min())}]).to_csv(tables / f"hier_diagnostics_{name}.csv", index=False)
        post = thin(idata)
        post.to_netcdf(path)

    preds = pd.read_parquet(root / f"data/processed/preds_{name}.parquet")
    y = y_show[te].values
    p_broken = predict(post, X.loc[te], g_te)
    preds["hierarchical Bayes"] = 1 - p_broken.mean(0)
    if name == "noshow":  # also use each patient's earlier visits in the test period, like the other models do
        preds["hierarchical Bayes, online"] = 1 - online(p_broken, 1 - y, people[te].values, df.loc[te, date_col].values)
    for model, p in preds.items():
        p = p.values
        vs_move = bootstrap_diff(y, p, preds["MOVE C"].values, people[te].values)
        vs_best = bootstrap_diff(y, p, preds[best].values, people[te].values)
        rows.append({"dataset": name, "model": model, "stratum": "all", **evaluate(y, p),
                     "vs_MOVE": vs_move[0], "vs_MOVE_ci": f"{vs_move[1]:.3f} to {vs_move[2]:.3f}",
                     "vs_best": vs_best[0], "vs_best_ci": f"{vs_best[1]:.3f} to {vs_best[2]:.3f}"})
        for s in strata.cat.categories:
            m = (strata[te] == s).values
            rows.append({"dataset": name, "model": model, "stratum": s, **evaluate(y[m], p[m])})

    out = pd.DataFrame(rows)
    out.to_csv(tables / f"rq2_comparison_{name}.csv", index=False)
    print(pd.read_csv(tables / f"hier_diagnostics_{name}.csv").to_string(index=False))
    print(out[out.stratum == "all"].drop(columns=["dataset", "stratum"]).round(3).to_string(index=False))
    print(out.pivot_table(index="model", columns="stratum", values="log_loss", sort=False).round(3).to_string())
