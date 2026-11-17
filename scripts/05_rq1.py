from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from thesis.data.datasets import noshow_data
from thesis.data.features import split
from thesis.models.hierarchical import codes, fit

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"

# no-show: same model as 04 but with past behaviour as features and no patient effect,
# so past behaviour gets its own effect estimate (the prediction model can't have both)
df = noshow_data()[0]
tr = split(df)[0].index
num = pd.DataFrame({"log_lead": np.log1p(df.lead_days), "age": df.age,
                    "log_prior_no_shows": np.log1p(df.prior_no_shows),
                    "log_prior_shows": np.log1p(df.prior_visits - df.prior_no_shows)})
binary = df[["scholarship", "hypertension", "diabetes", "alcoholism", "handicap", "sms_received"]].assign(
    female=(df.gender == "F").astype(int), same_day=(df.lead_days == 0).astype(int))
X = pd.concat([(num - num.loc[tr].mean()) / num.loc[tr].std(), binary], axis=1)
idata = fit(X.loc[tr], df.no_show[tr].values, {"neighbourhood": codes(df.neighbourhood[tr], df.neighbourhood[tr])[0]})
az.summary(idata, var_names=["a", "b", "sigma_neighbourhood"], ci_prob=0.9).to_csv(tables / "rq1_noshow_effects.csv")

# one row per MOVE driver, as odds ratios with 90% intervals. Numeric drivers are per 1 sd
noshow = pd.read_csv(tables / "rq1_noshow_effects.csv", index_col=0)
hotel = pd.read_csv(tables / "hier_effects_hotel.csv", index_col=0)
drivers = {"lead time": ("log_lead", "log_lead"), "past broken commitments": ("log_prior_no_shows", "log_prev_cancellations"),
           "past kept commitments": ("log_prior_shows", "log_prev_kept"), "same-day commitment": ("same_day", None),
           "reminder (SMS)": ("sms_received", None),
           "non-refundable deposit": (None, "Non Refund"), "price per night": (None, "adr"), "group size": (None, "group_size")}


def odds(table, feature):
    if feature is None:
        return np.nan, ""
    r = table.loc[f"b[{feature}]"]
    return r["mean"], f"{np.exp(r['mean']):.2f} ({np.exp(r.eti90_lb):.2f} to {np.exp(r.eti90_ub):.2f})"


rows = []
for driver, (f_noshow, f_hotel) in drivers.items():
    (b1, or1), (b2, or2) = odds(noshow, f_noshow), odds(hotel, f_hotel)
    same = "" if np.isnan(b1) or np.isnan(b2) else "yes" if np.sign(b1) == np.sign(b2) else "no"
    rows.append({"driver": driver, "no-show odds ratio": or1, "hotel odds ratio": or2, "same direction": same})
out = pd.DataFrame(rows)
out.to_csv(tables / "rq1_drivers.csv", index=False)
print(out.to_string(index=False))
