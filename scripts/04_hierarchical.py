import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

from thesis.data.datasets import hotel_data, hotel_design, noshow_data, noshow_design
from thesis.data.features import split
from thesis.eval.metrics import bootstrap_diff, evaluate
from thesis.models.hierarchical import codes, fit, online, predict, thin

root = Path(__file__).resolve().parent.parent

datasets = {"noshow": (noshow_data, noshow_design, "2016-05-25", "LightGBM + isotonic", {"draws": 2000}),
            "hotel": (hotel_data, hotel_design, "2016-12-31", "LightGBM", {"target_accept": 0.95, "tune": 2000})}
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
    preds.to_parquet(root / f"data/processed/preds_{name}_all.parquet")
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
