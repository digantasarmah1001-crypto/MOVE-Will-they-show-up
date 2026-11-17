from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from thesis.data.features import add_history, split
from thesis.models.move import fit_prior
from thesis.sim.gaming import personas, recovery, whitewash
from thesis.sim.platform import (beta_scorer, calibrate, invite_capacity, invite_fixed_ratio, invite_optimised,
                                 move_scorer, simulate)

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
seeds = range(5)
raw = pd.read_parquet(root / "data/processed/noshow.parquet")
mean_show = 1 - split(raw)[0].no_show.mean()
sigma = pd.read_csv(tables / "hier_effects_noshow.csv", index_col=0).loc["sigma_patient_id", "mean"]
a, b, population = calibrate(mean_show, sigma)
setup = dict(mean_show=mean_show, sigma=sigma)
scorers = {"MOVE C": lambda s: move_scorer, "learned": lambda s: beta_scorer(a, b, seed=s)}
rules = {"invite capacity": invite_capacity, "fixed ratio": invite_fixed_ratio(1 / mean_show), "optimised": invite_optimised}


def summary(plans):
    return {"value per plan": plans.value.mean(), "plans that happen": plans.happened.mean(),
            "turned away per plan": plans.overflow.mean(), "invited per plan": plans.invited.mean()}


def run(label, **kw):
    runs = pd.DataFrame([summary(simulate(**kw, **setup, seed=s)[0]) for s in seeds])
    return {**label, **runs.mean().to_dict(), "value sd over seeds": runs["value per plan"].std()}


def save(table, name):
    table.to_csv(tables / f"{name}.csv", index=False)
    print(table.round(3).to_string(index=False))


# invite rules and scorers
save(pd.DataFrame([run({"scorer": sn, "rule": rn}, scorer=sf(0), rule=rf)
                   for sn, sf in scorers.items() for rn, rf in rules.items()]), "sim_policies")
# fairness: 10% of users are flaky for 30 days, then reform
rows = []
for fair in [None, 7, 14]:
    for s in seeds:
        plans, hist, _, reformed = simulate(beta_scorer(a, b, seed=s), invite_optimised, reform=0.1, fair_days=fair,
                                            **setup, seed=s)
        later = np.array([sum(d >= 30 for d, _ in h) for h in hist])
        score = beta_scorer(a, b, seed=s)([h for h, r in zip(hist, reformed) if r], 89).mean(0)
        rows.append({"fair_days": fair or "off", **summary(plans), "invites per reformed user after day 30": later[reformed].mean(),
                     "invites per other user after day 30": later[~reformed].mean(), "reformed users' final score": score.mean()})
fairness = pd.DataFrame(rows).groupby("fair_days", sort=False).mean().reset_index()
fairness["price of fairness"] = 1 - fairness["value per plan"] / fairness["value per plan"].iloc[0]
save(fairness, "sim_fairness")
# whitewashing among the least reliable quarter, for lower newcomer starts with the same prior strength
_, hist, true_p, _ = simulate(beta_scorer(a, b), invite_optimised, **setup)
rows = []
for start in [a / (a + b), 0.75, 0.7, 0.6]:
    w = whitewash(hist, true_p, 89, start * (a + b), (1 - start) * (a + b))
    for sc, g in w[w.true_p < np.quantile(true_p, 0.25)].groupby("scorer"):
        if sc == "learned" or start == a / (a + b):
            rows.append({"scorer": sc, "newcomer start": 2 / 3 if sc == "MOVE C" else start,
                         "share of bottom 25% who gain": (g.gain > 0).mean(), "mean gain": g.gain.mean()})
gaming = pd.DataFrame(rows)
# genuine newcomers show up as often as everyone else, so a lower start costs them accuracy
gaming["log loss for a genuine newcomer"] = [-(population * np.log(x) + (1 - population) * np.log(1 - x)).mean()
                                             for x in gaming["newcomer start"]]
save(gaming, "sim_whitewash")
# recovery by half-life, and what each half-life costs in prediction on the real data
save(pd.DataFrame([{"half_life": hl, "target": t, **recovery(a, b, target=t, half_life=hl)}
                   for hl in [7, 14, 30] for t in [0.7, 0.75, 0.8]]), "sim_recovery")
cost = []
for hl in [7, 14, 30]:
    tr, te = split(add_history(raw, half_life=hl))
    pa, pb = fit_prior(tr.w_shows.values, tr.w_visits.values, 1 - tr.no_show.values)
    cost.append({"half_life": hl, "test log loss": log_loss(1 - te.no_show, (te.w_shows + pa) / (te.w_visits + pa + pb))})
save(pd.DataFrame(cost), "half_life_cost")
save(personas(a, b), "sim_personas")
# how bad a failed plan is
save(pd.DataFrame([run({"fail_cost": fc, "rule": rn}, scorer=beta_scorer(a, b), rule=rf,
                       costs={"reward": 1.0, "over_cost": 1.0, "fail_cost": fc})
                   for fc in [1.0, 2.0, 4.0] for rn, rf in rules.items()]), "sim_sensitivity")
