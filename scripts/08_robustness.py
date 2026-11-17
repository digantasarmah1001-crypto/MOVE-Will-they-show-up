from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from thesis.data.datasets import hotel_data, hotel_design
from thesis.data.features import split
from thesis.models.hierarchical import fit

root = Path(__file__).resolve().parent.parent
df = split(hotel_data()[0], "date", "2016-12-31")[0]

# do the hotel drivers hold up when we change the data or the outcome? Simple Bayesian logistic
# regressions (no group effects, so all four are comparable with each other)
late = lambda n: df.is_canceled.astype(bool) & (df.cancel_days_before_arrival <= n)
early = df.is_canceled.astype(bool) & ~late(7)
versions = {"all bookings": (df.index, df.is_canceled),
            "without non-refundable": (df.index[df.deposit_type != "Non Refund"], df.is_canceled),
            "late cancel, 7 days or less": (df.index[~early], late(7)),  # early cancels left out, like a harmless early cancel in MOVE
            "late cancel, 1 day or less": (df.index[~df.is_canceled.astype(bool) | late(1)], late(1))}

drivers = ["log_lead", "log_prev_cancellations", "log_prev_kept", "group_size", "adr", "Non Refund"]
rows = []
for name, (idx, y) in versions.items():
    num, binary, _ = hotel_design(df.loc[idx])
    X = pd.concat([(num - num.mean()) / num.std(), binary], axis=1).loc[:, lambda d: d.std() > 0]
    s = az.summary(fit(X, y.loc[idx].astype(int).values, {}, draws=500), var_names=["b"], ci_prob=0.9)
    for d in drivers:
        if f"b[{d}]" in s.index:
            r = s.loc[f"b[{d}]"]
            rows.append({"version": name, "rows": len(idx), "rate": y.loc[idx].mean(), "driver": d,
                         "odds ratio": np.exp(r["mean"]), "low": np.exp(r.eti90_lb), "high": np.exp(r.eti90_ub)})
out = pd.DataFrame(rows)
out.to_csv(root / "results/tables/robustness_hotel.csv", index=False)
print(out.pivot_table(index="driver", columns="version", values="odds ratio", sort=False).round(2).to_string())
print(out.groupby("version", sort=False)[["rows", "rate"]].first().round(3).to_string())
