from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

from thesis.data.datasets import noshow_data
from thesis.data.features import split

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
MODELS = ["MOVE C", "logistic regression", "LightGBM + isotonic", "hierarchical Bayes, online"]
THRESHOLDS = np.round(np.arange(0.05, 0.51, 0.01), 2)
df = noshow_data()["df"]
t = df.loc[split(df)[1].index].reset_index(drop=True)
no_show = t.no_show.values
risk = 1 - pd.read_parquet(root / "data/processed/preds_noshow_all.parquet").reset_index(drop=True)[MODELS]
# the overall level is re-estimated on the first test days and everything is judged on the later ones
early, late = (t.appointment_day <= "2016-05-31").values, (t.appointment_day > "2016-05-31").values


def recalibrate(r, target):
    # shift on the logit scale so the average risk matches the target rate
    shift = brentq(lambda s: expit(logit(np.clip(r, 1e-6, 1 - 1e-6)) + s).mean() - target, -5, 5)
    return expit(logit(np.clip(r, 1e-6, 1 - 1e-6)) + shift)


def net_benefit(r, y, threshold):
    remind = r >= threshold
    return ((remind & (y == 1)).sum() - (remind & (y == 0)).sum() * threshold / (1 - threshold)) / len(y)


def remind_everyone(y, threshold):
    return y.mean() - (1 - y.mean()) * threshold / (1 - threshold)


rows = []
y = no_show[late]
for threshold in THRESHOLDS:
    rows.append({"threshold": threshold, "strategy": "remind everyone", "version": "", "net benefit": remind_everyone(y, threshold)})
    rows.append({"threshold": threshold, "strategy": "remind no one", "version": "", "net benefit": 0.0})
    for model in MODELS:
        fixed = recalibrate(risk[model].values[late], no_show[early].mean())
        for version, r in [("as fitted", risk[model].values[late]), ("level updated", fixed)]:
            rows.append({"threshold": threshold, "strategy": model, "version": version, "net benefit": net_benefit(r, y, threshold)})
curves = pd.DataFrame(rows)
curves.to_csv(tables / "decision_curves.csv", index=False)
# patient bootstrap for the Bayesian model against the others, at a few thresholds
rows_of = t[late].reset_index(drop=True).groupby("patient_id").indices
patients = np.array(list(rows_of))
rng = np.random.default_rng(0)
r_late = {m: risk[m].values[late] for m in MODELS}
diffs = []
for _ in range(200):
    idx = np.concatenate([rows_of[p] for p in rng.choice(patients, len(patients))])
    for threshold in [0.1, 0.2, 0.3]:
        bayes = net_benefit(r_late["hierarchical Bayes, online"][idx], y[idx], threshold)
        others = {m: net_benefit(r_late[m][idx], y[idx], threshold) for m in MODELS[:3]}
        others["remind everyone"] = remind_everyone(y[idx], threshold)
        diffs += [{"threshold": threshold, "against": k, "difference": bayes - v} for k, v in others.items()]
diffs = pd.DataFrame(diffs).groupby(["threshold", "against"]).difference.quantile([0.025, 0.5, 0.975]).unstack()
diffs.columns = ["low", "median", "high"]
diffs.to_csv(tables / "decision_curve_differences.csv")
# no-shows prevented per 100 SMS-like reminders, only for bookings 3+ days ahead where the effect was measured.
# Three versions of the effect: the same for everyone, proportional to baseline risk, and as measured per risk third
by_risk = pd.read_csv(tables / "sms_effect_by_risk.csv").set_index("risk group")["absolute effect"]
overall = pd.read_csv(tables / "sms_effect.csv").set_index("estimate").loc["effect of one SMS (outage design)", "value"]
base = pd.read_parquet(root / "data/processed/baseline_risk.parquet")
e = t[late].reset_index(drop=True).merge(base, on="appointment_id", how="left")
eligible = ((e.lead_days >= 3) & e.baseline_risk.notna()).values
e = e[eligible].reset_index(drop=True)
rows = []
for share in [0.1, 0.2, 0.3, 0.5, 1.0]:
    for model in MODELS + ["random"]:
        r = np.random.default_rng(0).random(len(e)) if model == "random" else risk[model].values[late][eligible]
        chosen = e.iloc[np.argsort(-r)[:int(share * len(e))]]
        rows.append({"share reminded": share, "model": model, "no-shows among reminded": chosen.no_show.mean(),
                     "same for everyone": -100 * overall,
                     "proportional to risk": -100 * overall * chosen.baseline_risk.mean() / e.baseline_risk.mean(),
                     "measured per risk third": -100 * chosen.risk_group.map(by_risk).mean()})
prevented = pd.DataFrame(rows)
prevented.to_csv(tables / "reminders_prevented.csv", index=False)
# how coarse each score is: share of later visits whose exact score is shared with 50 or more other visits
ties = pd.DataFrame([{"model": m, "share with a tied score": (pd.Series(r_late[m]).map(pd.Series(r_late[m]).value_counts()) > 50).mean()}
                     for m in MODELS])
ties.to_csv(tables / "score_ties.csv", index=False)
show = curves[curves.threshold.isin([0.1, 0.2, 0.3, 0.4])].pivot_table(index=["strategy", "version"], columns="threshold",
                                                                       values="net benefit", sort=False)
print(f"later test days: {late.sum()} visits, no-show rate {y.mean():.3f}; first test days rate {no_show[early].mean():.3f}")
print(show.round(4).to_string())
print(diffs.round(4).to_string())
print(f"eligible later visits: {len(e)}")
print(prevented.round(2).to_string(index=False))
print(ties.round(3).to_string(index=False))
