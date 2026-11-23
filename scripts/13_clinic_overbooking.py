from pathlib import Path

import numpy as np
import pandas as pd

from thesis.data.datasets import noshow_data
from thesis.data.features import split
from thesis.opt.overbooking import best_extra, clinic_cost

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
df = noshow_data()["df"]
train, test = split(df)
t = test.reset_index(drop=True)
preds = pd.read_parquet(root / "data/processed/preds_noshow_all.parquet").reset_index(drop=True)
p = {"MOVE C": preds["MOVE C"].values, "hierarchical Bayes": preds["hierarchical Bayes, online"].values}
showed = 1 - t.no_show.values
train_show = 1 - train.no_show.mean()
# a clinic-day (area and day) has as many slots as it had bookings. Extra patients come from other real
# patients booked that day, in a random order shared by all policies, and bring their real outcome
sessions = [rows for rows in t.groupby(["neighbourhood", "appointment_day"]).indices.values() if len(rows) >= 5]
pool = t.groupby("appointment_day").indices
day_of = t.appointment_day.values
rows = []
burden = []
for over_cost in [0.5, 1.0, 2.0]:
    costs = {"idle_cost": 1.0, "over_cost": over_cost}
    decisions = {"no overbooking": [0] * len(sessions),
                 "fixed rate": [round(len(s) * (1 / train_show - 1)) for s in sessions]}
    for model, prob in p.items():
        decisions[model] = [best_extra(prob[s], prob[pool[day_of[s[0]]]].mean(), len(s), **costs) for s in sessions]
    for seed in range(10):
        rng = np.random.default_rng(seed)
        for i, s in enumerate(sessions):
            others = np.setdiff1d(pool[day_of[s[0]]], s)
            extras = rng.permutation(others)
            booked_shows = showed[s].sum()
            real = clinic_cost(booked_shows + np.cumsum(np.append(0, showed[extras])), len(s), **costs)
            for policy, ks in decisions.items():
                k = min(ks[i], len(extras))
                shows = booked_shows + showed[extras[:k]].sum()
                rows.append({"overflow cost": over_cost, "seed": seed, "policy": policy, "extra booked": k,
                             "idle slots": max(len(s) - shows, 0), "overflow": max(shows - len(s), 0), "cost": real[k]})
                if over_cost == 1.0:
                    came = s[showed[s] == 1]
                    burden += [{"seed": seed, "policy": policy, "welfare": w, "overflow per patient": max(shows - len(s), 0) / shows}
                               for w in t.scholarship.values[came]]
            rows.append({"overflow cost": over_cost, "seed": seed, "policy": "perfect foresight",
                         "extra booked": int(np.argmin(real)), "cost": real.min()})
res = pd.DataFrame(rows)
summary = res.groupby(["overflow cost", "policy"], sort=False).agg(
    cost=("cost", "mean"), extra_booked=("extra booked", "mean"), idle_slots=("idle slots", "mean"),
    overflow=("overflow", "mean"), clinic_days_with_overflow=("overflow", lambda o: (o > 0).mean())).reset_index()
sd = res.groupby(["overflow cost", "policy", "seed"], sort=False).cost.mean().groupby(["overflow cost", "policy"], sort=False).std()
summary["cost sd over seeds"] = sd.values
summary.to_csv(tables / "clinic_policies.csv", index=False)
fair = pd.DataFrame(burden).groupby(["policy", "welfare"], sort=False)["overflow per patient"].mean().unstack()
fair.columns = ["not on welfare", "on welfare"]
fair.to_csv(tables / "clinic_fairness.csv")
print(f"{len(sessions)} clinic-days, {sum(map(len, sessions))} booked visits")
print(summary.round(3).to_string(index=False))
print(fair.round(4).to_string())
