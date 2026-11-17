from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from thesis.data.features import split
from thesis.opt.matching import match_greedy, match_ilp, simulate_matching
from thesis.sim.platform import beta_scorer, calibrate

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
mean_show = 1 - split(pd.read_parquet(root / "data/processed/noshow.parquet"))[0].no_show.mean()
sigma = pd.read_csv(tables / "hier_effects_noshow.csv", index_col=0).loc["sigma_patient_id", "mean"]
a, b, _ = calibrate(mean_show, sigma)
rows = []
for available in [60, 90, 120]:
    for name, match in [("greedy", match_greedy), ("integer programme", match_ilp)]:
        for seed in range(3):
            true_p = expit(logit(mean_show) + sigma * np.random.default_rng(seed).standard_normal(500))
            r = pd.DataFrame(simulate_matching(beta_scorer(a, b, seed=seed), match, true_p=true_p, days=30,
                                               available=available, seed=seed))
            rows.append({"available users": available, "method": name, "plans that happen": r.happened.mean(),
                         "value per plan": r.value.mean(), "invited per plan": r.invited.mean()})
out = pd.DataFrame(rows).groupby(["available users", "method"], sort=False).mean().reset_index()
out.to_csv(tables / "sim_matching.csv", index=False)
print(out.round(3).to_string(index=False))
