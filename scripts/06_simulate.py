from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from sklearn.metrics import log_loss

from thesis.data.features import add_history, split
from thesis.models.move import fit_prior
from thesis.sim.gaming import personas, recovery, whitewash
from thesis.sim.platform import (beta_scorer, invite_capacity, invite_fixed_ratio, invite_optimised, move_scorer,
                                 simulate)

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
seeds = range(5)

# people differ in reliability as the no-show model found: spread sigma_patient on the logit scale,
# average show rate as in the training data
effects = pd.read_csv(tables / "hier_effects_noshow.csv", index_col=0)
sigma = effects.loc["sigma_patient_id", "mean"]
mean_show = 1 - pd.read_parquet(root / "data/processed/noshow.parquet").query("appointment_day <= '2016-05-25'").no_show.mean()
logit_mean = np.log(mean_show / (1 - mean_show))
population = expit(logit_mean + sigma * np.random.default_rng(0).standard_normal(100_000))
# learned prior = Beta with the population's mean and variance (what enough data would teach the scorer)
m, v = population.mean(), population.var()
a, b = m * (m * (1 - m) / v - 1), (1 - m) * (m * (1 - m) / v - 1)
setup = dict(mean_show=mean_show, sigma=sigma)
print(f"mean show {mean_show:.3f}, sigma {sigma:.2f}, prior Beta({a:.2f}, {b:.2f})")

scorers = {"MOVE C": lambda s: move_scorer, "learned": lambda s: beta_scorer(a, b, seed=s)}
rules = {"invite capacity": invite_capacity, "fixed ratio": invite_fixed_ratio(1 / mean_show), "optimised": invite_optimised}


def summary(plans):
    return {"value per plan": plans.value.mean(), "plans that happen": plans.happened.mean(),
            "turned away per plan": plans.overflow.mean(), "invited per plan": plans.invited.mean()}


def run(label, seeds=seeds, **kw):
    runs = pd.DataFrame([summary(simulate(**kw, **setup, seed=s)[0]) for s in seeds])
    return {**label, **runs.mean().to_dict(), "value sd over seeds": runs["value per plan"].std()}


# 1. invite rules and scorers
policies = pd.DataFrame([run({"scorer": sn, "rule": rn}, scorer=sf(0), rule=rf)
                         for sn, sf in scorers.items() for rn, rf in rules.items()])
policies.to_csv(tables / "sim_policies.csv", index=False)
print(policies.round(3).to_string(index=False))

# 2. fairness: 10% of users are flaky for the first 30 days, then reform
rows = []
for fair in [None, 7, 14]:
    for s in seeds:
        plans, hist, _, reformed = simulate(beta_scorer(a, b, seed=s), invite_optimised, reform=0.1, fair_days=fair, **setup, seed=s)
        later = np.array([sum(d >= 30 for d, _ in h) for h in hist])
        score = beta_scorer(a, b, seed=s)([h for h, r in zip(hist, reformed) if r], 89).mean(0)
        rows.append({"fair_days": fair or "off", **summary(plans), "invites per reformed user after day 30": later[reformed].mean(),
                     "invites per other user after day 30": later[~reformed].mean(), "reformed users' final score": score.mean()})
fairness = pd.DataFrame(rows).groupby("fair_days", sort=False).mean().reset_index()
off = fairness.loc[fairness.fair_days == "off", "value per plan"].iloc[0]
fairness["price of fairness"] = 1 - fairness["value per plan"] / off
fairness.to_csv(tables / "sim_fairness.csv", index=False)
print(fairness.round(3).to_string(index=False))

# 3. whitewashing, and what a lower starting score for newcomers costs genuine newcomers
_, hist, true_p, _ = simulate(beta_scorer(a, b), invite_optimised, **setup)
rows = []
for start in [m, 0.75, 0.7, 0.6]:
    n = a + b  # same prior strength, lower mean
    w = whitewash(hist, true_p, 89, start * n, (1 - start) * n)
    w = w[w.true_p < np.quantile(true_p, 0.25)]  # the least reliable quarter
    for sc, g in w.groupby("scorer"):
        if sc == "learned" or start == m:  # MOVE C has its own fixed start, so it only needs one row
            rows.append({"scorer": sc, "newcomer start": 2 / 3 if sc == "MOVE C" else start,
                         "share of bottom 25% who gain": (g.gain > 0).mean(), "mean gain": g.gain.mean()})
gaming = pd.DataFrame(rows)
# a lower start is wrong for genuine newcomers, who show up as often as everyone else
gaming["log loss for a genuine newcomer"] = [-(population * np.log(x) + (1 - population) * np.log(1 - x)).mean()
                                             for x in gaming["newcomer start"]]
gaming.to_csv(tables / "sim_whitewash.csv", index=False)
print(gaming.round(3).to_string(index=False))

# 4. recovery and 5. personas
# the data can't tell half-lives apart (6 week window), so the half-life is a product choice: show what it does
rec = pd.DataFrame([{"half_life": hl, "target": t, **recovery(a, b, target=t, half_life=hl)}
                    for hl in [7, 14, 30] for t in [0.7, 0.75, 0.8]])
rec.to_csv(tables / "sim_recovery.csv", index=False)
# and what each half-life costs in prediction on the real no-show data (tuned MOVE C, test period)
raw = pd.read_parquet(root / "data/processed/noshow.parquet")
cost = []
for hl in [7, 14, 30]:
    tr, te = split(add_history(raw, half_life=hl))
    pa, pb = fit_prior(tr.w_shows.values, tr.w_visits.values, 1 - tr.no_show.values)
    cost.append({"half_life": hl, "test log loss": log_loss(1 - te.no_show, (te.w_shows + pa) / (te.w_visits + pa + pb))})
pd.DataFrame(cost).to_csv(tables / "half_life_cost.csv", index=False)
print(pd.DataFrame(cost).round(4).to_string(index=False))
per = personas(a, b)
per.to_csv(tables / "sim_personas.csv", index=False)
print(rec.to_string(index=False))
print(per.round(2).to_string(index=False))

# 6. sensitivity to how bad a failed plan is
sens = pd.DataFrame([run({"fail_cost": fc, "rule": rn}, scorer=beta_scorer(a, b), rule=rf,
                         costs={"reward": 1.0, "over_cost": 1.0, "fail_cost": fc})
                     for fc in [1.0, 2.0, 4.0] for rn, rf in rules.items()])
sens.to_csv(tables / "sim_sensitivity.csv", index=False)
print(sens.round(3).to_string(index=False))
