import sys
from pathlib import Path

import pandas as pd
import xarray as xr

from thesis.data.datasets import hotel_data, hotel_design, noshow_data, noshow_design
from thesis.data.features import split
from thesis.eval.metrics import bootstrap_diff, evaluate
from thesis.models.hierarchical import codes, fit, online, predict, summarise, thin

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
datasets = {"noshow": (noshow_data, noshow_design, "2016-05-25", "LightGBM + isotonic", {"draws": 2000}),
            "hotel": (hotel_data, hotel_design, "2016-12-31", "LightGBM", {"target_accept": 0.95, "tune": 2000})}
for name in sys.argv[1:] or datasets:
    load, design, split_date, best, sampler = datasets[name]
    d = load()
    df, y, people = d["df"], d["y"], d["people"]
    tr, te = (part.index for part in split(df, d["date"], split_date))
    num, binary, group_cols = design(df)
    X = pd.concat([(num - num.loc[tr].mean()) / num.loc[tr].std(), binary], axis=1)
    g_tr, g_te = {}, {}
    for g in group_cols:
        g_tr[g], g_te[g] = codes(df.loc[tr, g], df.loc[te, g])
    # a saved fit is reused. Delete results/models/hier_<name>.nc to refit
    path = root / f"results/models/hier_{name}.nc"
    if not path.exists():
        idata = fit(X.loc[tr], 1 - y[tr].values, g_tr, **sampler)
        effects, checks = summarise(idata, group_cols)
        effects.to_csv(tables / f"hier_effects_{name}.csv")
        checks.to_csv(tables / f"hier_diagnostics_{name}.csv", index=False)
        thin(idata).to_netcdf(path)
    p_broken = predict(xr.open_dataset(path), X.loc[te], g_te)
    preds = pd.read_parquet(root / f"data/processed/preds_{name}.parquet")
    preds["hierarchical Bayes"] = 1 - p_broken.mean(0)
    if name == "noshow":
        preds["hierarchical Bayes, online"] = 1 - online(p_broken, 1 - y[te].values, people[te].values,
                                                         df.loc[te, d["date"]].values)
    preds.to_parquet(root / f"data/processed/preds_{name}_all.parquet")
    rows = []
    for model, p in preds.items():
        p, yt, who = p.values, y[te].values, people[te].values
        vs_move = bootstrap_diff(yt, p, preds["MOVE C"].values, who)
        vs_best = bootstrap_diff(yt, p, preds[best].values, who)
        rows.append({"dataset": name, "model": model, "stratum": "all", **evaluate(yt, p),
                     "vs_MOVE": vs_move[0], "vs_MOVE_ci": f"{vs_move[1]:.3f} to {vs_move[2]:.3f}",
                     "vs_best": vs_best[0], "vs_best_ci": f"{vs_best[1]:.3f} to {vs_best[2]:.3f}"})
        for s in d["strata"].cat.categories:
            m = (d["strata"][te] == s).values
            rows.append({"dataset": name, "model": model, "stratum": s, **evaluate(yt[m], p[m])})
    out = pd.DataFrame(rows)
    out.to_csv(tables / f"rq2_comparison_{name}.csv", index=False)
    print(pd.read_csv(tables / f"hier_diagnostics_{name}.csv").to_string(index=False))
    print(out[out.stratum == "all"].drop(columns=["dataset", "stratum"]).round(3).to_string(index=False))
