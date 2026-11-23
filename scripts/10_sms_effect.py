from pathlib import Path

import numpy as np
import pandas as pd

from thesis.data.datasets import noshow_data
from thesis.models.causal import balance, diff_in_diff, e_value, effects, propensity

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
# the SMS system sent nothing on these appointment days, while on other days about 85% of eligible visits got one
OUTAGE = pd.to_datetime(["2016-05-09", "2016-05-13", "2016-05-16", "2016-05-17", "2016-05-18", "2016-05-19", "2016-05-20"])
df = noshow_data()["df"]
df = df[~df.sms_mixed].reset_index(drop=True)
df["eligible"] = (df.lead_days >= 3).astype(int)
df["outage"] = df.appointment_day.isin(OUTAGE).astype(int)
person = pd.DataFrame({"age": df.age, "female": (df.gender == "F").astype(int), "log_lead": np.log1p(df.lead_days),
                       "log_prior_visits": np.log1p(df.prior_visits), "log_prior_no_shows": np.log1p(df.prior_no_shows)})
person = person.join(df[["scholarship", "hypertension", "diabetes", "alcoholism", "handicap"]])
covariates = person.join(pd.get_dummies(df.appointment_day.dt.day_name(), prefix="day", drop_first=True, dtype=int))


def outage_effect(rows):
    # difference in differences: how much more the eligible group's no-show rises on outage days than the
    # never-eligible group's. Divided by the drop in SMS it gives the effect of one SMS
    X, d = covariates.loc[rows].reset_index(drop=True), df.loc[rows].reset_index(drop=True)
    on_no_show = diff_in_diff(X, d.no_show.values, d.eligible.values, d.outage.values)
    on_sms = diff_in_diff(X, d.sms_received.values, d.eligible.values, d.outage.values)
    return on_no_show, on_no_show / on_sms


def propensity_effect(rows):
    # the usual comparison of patients with and without an SMS, on normal days only (where both exist)
    d = df.loc[rows]
    keep = (d.eligible == 1) & (d.outage == 0)
    X = person.loc[d.index[keep]].reset_index(drop=True)
    return effects(X, d.sms_received.values[keep], d.no_show.values[keep])


everyone = df.index.values
did, per_sms = outage_effect(everyone)
prop = propensity_effect(everyone)
rng = np.random.default_rng(0)
rows_of = df.groupby("patient_id").indices
patients = np.array(list(rows_of))
boot = []
for _ in range(200):
    rows = np.concatenate([rows_of[p] for p in rng.choice(patients, len(patients))])
    b_did, b_sms = outage_effect(rows)
    b = propensity_effect(rows)
    boot.append([b_did, b_sms, b["weighting"], b["regression"], b["doubly robust"], b["risk with"] / b["risk without"]])
low, high = np.percentile(boot, [2.5, 97.5], axis=0)
point = [did, per_sms, prop["weighting"], prop["regression"], prop["doubly robust"], prop["risk with"] / prop["risk without"]]
names = ["outage raises no-show by", "effect of one SMS (outage design)", "effect of SMS, weighting (normal days)",
         "effect of SMS, regression (normal days)", "effect of SMS, doubly robust (normal days)", "risk ratio, regression (normal days)"]
out = pd.DataFrame({"estimate": names, "value": point, "low": low, "high": high})
rr = point[-1]
out["e_value"] = [np.nan] * 5 + [e_value(rr)]
out["e_value_ci"] = [np.nan] * 5 + [e_value(high[-1]) if high[-1] < 1 else 1.0]
out.to_csv(tables / "sms_effect.csv", index=False)
# checks: the never-eligible group should not change on outage days, and the groups should be balanced
check = df.groupby(["eligible", "outage"]).agg(visits=("no_show", "size"), no_show=("no_show", "mean"),
                                               sms=("sms_received", "mean")).reset_index()
check.to_csv(tables / "sms_outage_check.csv", index=False)
normal = df.index[(df.eligible == 1) & (df.outage == 0)]
t = df.sms_received.values[normal]
ps = propensity(person.loc[normal], t)
w = np.where(t == 1, 1 / ps, 1 / (1 - ps))
bal = pd.DataFrame({"before": balance(person.loc[normal], t), "after weighting": balance(person.loc[normal], t, w)})
bal.to_csv(tables / "sms_balance.csv")
print(check.round(4).to_string(index=False))
print(out.round(4).to_string(index=False))
print(f"propensity range on normal days {ps.min():.3f} to {ps.max():.3f}, largest imbalance", bal.abs().max().round(3).to_dict())
