import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit, logsumexp


def codes(train, test):
    # training levels first, then levels only seen in test (new patients), so new ones have codes >= n_train
    cats = pd.Categorical(train).categories
    cats = cats.append(pd.Index(pd.unique(test)).difference(cats))
    return pd.Categorical(train, cats).codes, pd.Categorical(test, cats).codes


def fit(X, y, groups, draws=1000, tune=1000, target_accept=0.8, seed=0):
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
        return pm.sample(draws, tune=tune, chains=4, nuts_sampler="nutpie", target_accept=target_accept, random_seed=seed, progressbar=False)


def thin(idata, n_draws=500):
    # keep n_draws posterior draws as float32, enough for prediction and much smaller on disk
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    return post.isel(s=np.linspace(0, post.sizes["s"] - 1, n_draws).astype(int)).astype("float32").reset_index("s", drop=True)


def predict(post, X, groups, seed=0):
    # P(broken) for every posterior draw, shape (draws, rows). A new level (code >= the number of
    # trained levels) gets its own random effect drawn from N(0, sigma) in each draw
    rng = np.random.default_rng(seed)
    eta = post.a.values[:, None] + post.b.values.T @ X.values.T
    for g, idx in groups.items():
        z = post[f"z_{g}"].values.T
        new = rng.standard_normal((z.shape[0], max(int(idx.max()) + 1 - z.shape[1], 0)))
        eta += post[f"sigma_{g}"].values[:, None] * np.hstack([z, new])[:, idx]
    return expit(eta)


def online(p, y, person, day):
    # update on each person's own earlier outcomes: reweight the posterior draws by how well they
    # predicted those outcomes, then average. Rows must be one per person per day
    order = np.lexsort((day, person))
    ll = np.where(y, np.log(p), np.log1p(-p))[:, order]
    before = np.cumsum(ll, axis=1) - ll  # log likelihood of all earlier rows ...
    start = np.r_[0, np.flatnonzero(np.diff(person[order])) + 1]
    first = np.repeat(start, np.diff(np.r_[start, len(order)]))
    before -= np.cumsum(ll, axis=1)[:, first] - ll[:, first]  # ... of the same person only
    w = np.exp(before - logsumexp(before, axis=0))
    out = np.empty(p.shape[1])
    out[order] = (w * p[:, order]).sum(0)
    return out
