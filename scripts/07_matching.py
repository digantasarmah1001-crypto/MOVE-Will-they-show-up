from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from thesis.opt.matching import match_greedy, match_ilp, simulate_matching
from thesis.sim.platform import beta_scorer

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"

# same population and learned prior as 06_simulate.py
sigma = pd.read_csv(tables / "hier_effects_noshow.csv", index_col=0).loc["sigma_patient_id", "mean"]
mean_show = 1 - pd.read_parquet(root / "data/processed/noshow.parquet").query("appointment_day <= '2016-05-25'").no_show.mean()
population = expit(np.log(mean_show / (1 - mean_show)) + sigma * np.random.default_rng(0).standard_normal(100_000))
m, v = population.mean(), population.var()
a, b = m * (m * (1 - m) / v - 1), (1 - m) * (m * (1 - m) / v - 1)

rows = []
for available in [60, 90, 120]:  # users available per day for 10 plans: scarce to plenty
    for name, match in [("greedy", match_greedy), ("integer programme", match_ilp)]:
        for seed in range(3):
            true_p = expit(np.log(mean_show / (1 - mean_show)) + sigma * np.random.default_rng(seed).standard_normal(500))
            r = pd.DataFrame(simulate_matching(beta_scorer(a, b, seed=seed), match, true_p=true_p, days=30,
                                               available=available, seed=seed))
            rows.append({"available users": available, "method": name, "plans that happen": r.happened.mean(),
                         "value per plan": r.value.mean(), "invited per plan": r.invited.mean()})
out = pd.DataFrame(rows).groupby(["available users", "method"], sort=False).mean().reset_index()
out.to_csv(tables / "sim_matching.csv", index=False)
print(out.round(3).to_string(index=False))
