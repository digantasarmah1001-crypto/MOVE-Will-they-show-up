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
    call = r >= threshold
    return ((call & (y == 1)).sum() - (call & (y == 0)).sum() * threshold / (1 - threshold)) / len(y)


rows = []
y = no_show[late]
for threshold in THRESHOLDS:
    rows.append({"threshold": threshold, "strategy": "call everyone", "version": "",
                 "net benefit": y.mean() - (1 - y.mean()) * threshold / (1 - threshold)})
    rows.append({"threshold": threshold, "strategy": "call no one", "version": "", "net benefit": 0.0})
    for model in MODELS:
        fixed = recalibrate(risk[model].values[late], no_show[early].mean())
        for version, r in [("as fitted", risk[model].values[late]), ("level updated", fixed)]:
            rows.append({"threshold": threshold, "strategy": model, "version": version, "net benefit": net_benefit(r, y, threshold)})
curves = pd.DataFrame(rows)
curves.to_csv(tables / "decision_curves.csv", index=False)
# patient bootstrap for the Bayesian model against the others, at a few thresholds
lt = t[late].reset_index(drop=True)
rows_of = lt.groupby("patient_id").indices
patients = np.array(list(rows_of))
rng = np.random.default_rng(0)
r_late = {m: risk[m].values[late] for m in MODELS}
diffs = []
for _ in range(200):
    idx = np.concatenate([rows_of[p] for p in rng.choice(patients, len(patients))])
    for threshold in [0.1, 0.2, 0.3]:
        bayes = net_benefit(r_late["hierarchical Bayes, online"][idx], y[idx], threshold)
        everyone = y[idx].mean() - (1 - y[idx].mean()) * threshold / (1 - threshold)
        others = {m: net_benefit(r_late[m][idx], y[idx], threshold) for m in MODELS[:3]}
        others["call everyone"] = everyone
        diffs += [{"threshold": threshold, "against": k, "difference": bayes - v} for k, v in others.items()]
diffs = pd.DataFrame(diffs).groupby(["threshold", "against"]).difference.quantile([0.025, 0.5, 0.975]).unstack()
diffs.columns = ["low", "median", "high"]
diffs.to_csv(tables / "decision_curve_differences.csv")
# no-shows prevented per 100 calls when calling the riskiest share of patients, if a reminder lowers each
# patient's no-show risk by the same proportion as an SMS did in the outage analysis
effect = pd.read_csv(tables / "sms_effect.csv").set_index("estimate").loc["effect of one SMS (outage design)", "value"]
check = pd.read_csv(tables / "sms_outage_check.csv")
without = check.query("eligible == 1 and outage == 1").no_show.iloc[0]
rr = (without + effect) / without
rows = []
for share in [0.1, 0.2, 0.3, 0.5, 1.0]:
    for model in MODELS + ["random"]:
        r = np.random.default_rng(0).random(late.sum()) if model == "random" else risk[model].values[late]
        called = np.argsort(-r)[:int(share * late.sum())]
        rows.append({"share called": share, "model": model, "no-shows among called": y[called].mean(),
                     "prevented per 100 calls": 100 * y[called].mean() * (1 - rr)})
prevented = pd.DataFrame(rows)
prevented.to_csv(tables / "reminders_prevented.csv", index=False)
show = curves[curves.threshold.isin([0.1, 0.2, 0.3, 0.4])].pivot_table(index=["strategy", "version"], columns="threshold",
                                                                       values="net benefit", sort=False)
print(f"later test days: {late.sum()} visits, no-show rate {y.mean():.3f}; first test days rate {no_show[early].mean():.3f}")
print(show.round(4).to_string())
print("hierarchical Bayes minus each alternative (net benefit, 95% interval)")
print(diffs.round(4).to_string())
print(f"assumed risk ratio of a reminder {rr:.3f}")
print(prevented.pivot_table(index="model", columns="share called", values="prevented per 100 calls", sort=False).round(2).to_string())
