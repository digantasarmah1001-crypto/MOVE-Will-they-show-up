import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit


def codes(train, test):
    # integer codes from the training levels; levels never seen in training get -1
    cats = pd.Categorical(train).categories
    return pd.Categorical(train, cats).codes, pd.Categorical(test, cats).codes


def fit(X, y, groups, draws=1000, tune=1000, seed=0):
    # logit P(broken) = a + X b + one random effect per grouping (non-centred: sigma * z)
    with pm.Model(coords={"feature": X.columns}):
        a = pm.Normal("a", 0, 1.5)
        b = pm.Normal("b", 0, 1, dims="feature")
        eta = a + X.values @ b
        for g, idx in groups.items():
            sigma = pm.HalfNormal(f"sigma_{g}", 1)
            z = pm.Normal(f"z_{g}", 0, 1, shape=int(idx.max()) + 1)
            eta = eta + sigma * z[idx]
        pm.Bernoulli("y", logit_p=eta, observed=y)
        return pm.sample(draws, tune=tune, chains=4, nuts_sampler="nutpie", random_seed=seed, progressbar=False)


def predict(idata, X, groups, n_draws=500, seed=0):
    # average P(broken) over posterior draws. A level not seen in training (a new patient)
    # gets a random effect drawn from N(0, sigma), so we average over what we don't know about them
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    rng = np.random.default_rng(seed)
    pick = rng.choice(post.sizes["s"], min(n_draws, post.sizes["s"]), replace=False)
    eta = post.a.values[pick][:, None] + post.b.values[:, pick].T @ X.values.T
    for g, idx in groups.items():
        z = post[f"z_{g}"].values[:, pick].T[:, np.maximum(idx, 0)]
        z = np.where(idx >= 0, z, rng.standard_normal(z.shape))
        eta += post[f"sigma_{g}"].values[pick][:, None] * z
    return expit(eta).mean(0)
