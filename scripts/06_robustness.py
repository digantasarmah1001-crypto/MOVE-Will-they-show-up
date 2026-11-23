from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from thesis.data.datasets import hotel_data, hotel_design
from thesis.data.features import split
from thesis.models.hierarchical import fit

root = Path(__file__).resolve().parent.parent
df = split(hotel_data()["df"], "arrival_date", "2016-12-31")[0]
cancelled = df.is_canceled.astype(bool)


def late(days):
    return cancelled & (df.cancel_days_before_arrival <= days)


# the hotel drivers with other data or outcomes. No group effects, so the four are comparable with each other.
# For late cancels the early cancels are left out, like a harmless early cancel in MOVE
versions = {"all bookings": (df.index, df.is_canceled),
            "without non-refundable": (df.index[df.deposit_type != "Non Refund"], df.is_canceled),
            "late cancel, 7 days or less": (df.index[~cancelled | late(7)], late(7)),
            "late cancel, 1 day or less": (df.index[~cancelled | late(1)], late(1))}
drivers = ["log_lead", "log_prev_cancellations", "log_prev_kept", "group_size", "adr", "Non Refund"]
rows = []
for name, (idx, y) in versions.items():
    num, binary, _ = hotel_design(df.loc[idx])
    X = pd.concat([(num - num.mean()) / num.std(), binary], axis=1)
    X = X.loc[:, X.std() > 0]
    s = az.summary(fit(X, y.loc[idx].astype(int).values, {}, draws=500), var_names=["b"], ci_prob=0.9)
    for d in drivers:
        if f"b[{d}]" in s.index:
            r = s.loc[f"b[{d}]"]
            rows.append({"version": name, "rows": len(idx), "rate": y.loc[idx].mean(), "driver": d,
                         "odds ratio": np.exp(r["mean"]), "low": np.exp(r.eti90_lb), "high": np.exp(r.eti90_ub)})
out = pd.DataFrame(rows)
out.to_csv(root / "results/tables/robustness_hotel.csv", index=False)
print(out.pivot_table(index="driver", columns="version", values="odds ratio", sort=False).round(2).to_string())
