import arviz as az
import numpy as np
import pandas as pd
import pymc as pm
from scipy.special import expit


def codes(train, test):
    # levels seen only in test (new patients) get codes after all training levels
    cats = pd.Categorical(train).categories
    cats = cats.append(pd.Index(pd.unique(test)).difference(cats))
    return pd.Categorical(train, cats).codes, pd.Categorical(test, cats).codes


def fit(X, y, groups, draws=1000, tune=1000, target_accept=0.8, seed=0):
    with pm.Model(coords={"feature": X.columns}):
        a = pm.Normal("a", 0, 1.5)
        b = pm.Normal("b", 0, 1, dims="feature")
        eta = a + X.values @ b
        for g, idx in groups.items():
            sigma = pm.HalfNormal(f"sigma_{g}", 1)
            z = pm.Normal(f"z_{g}", 0, 1, shape=int(idx.max()) + 1)
            eta = eta + sigma * z[idx]
        pm.Bernoulli("y", logit_p=eta, observed=y)
        return pm.sample(draws, tune=tune, chains=4, nuts_sampler="nutpie", target_accept=target_accept,
                         random_seed=seed, progressbar=False)


def thin(idata, n_draws=500):
    post = idata.posterior.to_dataset().stack(s=("chain", "draw"))
    keep = np.linspace(0, post.sizes["s"] - 1, n_draws).astype(int)
    return post.isel(s=keep).astype("float32").reset_index("s", drop=True)


def predict(post, X, groups, seed=0):
    # returns P(broken) per posterior draw and row. New levels get a random effect from N(0, sigma)
    rng = np.random.default_rng(seed)
    eta = post.a.values[:, None] + post.b.values.T @ X.values.T
    for g, idx in groups.items():
        z = post[f"z_{g}"].values.T
        n_new = int(idx.max()) + 1 - z.shape[1]
        if n_new > 0:
            z = np.hstack([z, rng.standard_normal((len(z), n_new))])
        eta += post[f"sigma_{g}"].values[:, None] * z[:, idx]
    return expit(eta)


def online(p, y, person, day):
    # each person's earlier outcomes reweight the posterior draws before their next prediction
    out = np.empty(p.shape[1])
    order = pd.DataFrame({"person": person, "day": day}).sort_values("day")
    for rows in order.groupby("person").groups.values():
        loglik = np.zeros(p.shape[0])
        for i in rows:
            w = np.exp(loglik - loglik.max())
            out[i] = (w * p[:, i]).sum() / w.sum()
            loglik += np.log(p[:, i]) if y[i] else np.log1p(-p[:, i])
    return out


def summarise(idata, groups):
    # effect estimates with 90% intervals, and the sampler checks over every parameter
    effects = az.summary(idata, var_names=["a", "b"] + [f"sigma_{g}" for g in groups], ci_prob=0.9)
    z = az.summary(idata, var_names=[f"z_{g}" for g in groups], kind="diagnostics")
    checks = pd.DataFrame([{"divergences": int(idata.sample_stats.diverging.sum()),
                            "max_r_hat": max(effects.r_hat.max(), z.r_hat.max()),
                            "min_ess_bulk": min(effects.ess_bulk.min(), z.ess_bulk.min())}])
    return effects, checks
