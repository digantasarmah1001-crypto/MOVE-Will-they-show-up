from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from thesis.data.datasets import hotel_data, noshow_data
from thesis.data.features import split

root = Path(__file__).resolve().parent.parent
tables, figs = root / "results/tables", root / "results/figures"
# validated colour-blind safe palette, fixed order: each model or series keeps its colour in every figure
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "font.size": 10})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(figs / f"{name}.png", dpi=200)
    plt.close(fig)


# 1. reliability diagrams on the test period
models = {"hierarchical Bayes": (BLUE, "o"), "MOVE C": (ORANGE, "s"), "logistic regression": (AQUA, "^"), "LightGBM": (YELLOW, "D")}
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, (name, load, date) in zip(axes, [("no-show", noshow_data, "2016-05-25"), ("hotel", hotel_data, "2016-12-31")]):
    df, _, y, *_, date_col = load()
    te = split(df, date_col, date)[1].index
    preds = pd.read_parquet(root / f"data/processed/preds_{'noshow' if name == 'no-show' else 'hotel'}_all.parquet")
    for model, (color, marker) in models.items():
        col = "hierarchical Bayes, online" if model == "hierarchical Bayes" and name == "no-show" else model
        observed, predicted = calibration_curve(y[te], preds[col], n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, color=color, marker=marker, label=model)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    ax.set(title=f"{name} data", xlabel="predicted chance of showing up", ylabel="observed share who showed up",
           xlim=(0.3, 1), ylim=(0.3, 1))
axes[0].legend(loc="upper left")
save(fig, "reliability")

# 2. drivers of broken commitments, odds ratios with 90% intervals
noshow = pd.read_csv(tables / "rq1_noshow_effects.csv", index_col=0)
hotel = pd.read_csv(tables / "hier_effects_hotel.csv", index_col=0)
drivers = [("same-day commitment", "same_day", None), ("lead time (per 1 sd)", "log_lead", "log_lead"),
           ("past broken commitments", "log_prior_no_shows", "log_prev_cancellations"),
           ("past kept commitments", "log_prior_shows", "log_prev_kept"), ("SMS reminder", "sms_received", None),
           ("group size", None, "group_size"), ("price per night", None, "adr")]
fig, ax = plt.subplots(figsize=(7, 4.4))
for k, (table, label, color, marker, shift) in enumerate([(noshow, "no-show data", BLUE, "o", -0.12), (hotel, "hotel data", ORANGE, "s", 0.12)]):
    rows = [(row, table.loc[f"b[{f[k]}]"]) for row, (_, *f) in enumerate(drivers) if f[k]]
    for row, r in rows:
        ax.plot(np.exp([r.eti90_lb, r.eti90_ub]), [row + shift] * 2, color=color)
    ax.plot([np.exp(r["mean"]) for _, r in rows], [row + shift for row, _ in rows], marker=marker, color=color,
            linestyle="none", label=label)
ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
ax.set(xscale="log", yticks=range(len(drivers)), yticklabels=[d[0] for d in drivers],
       xlabel="odds ratio of breaking the commitment (log scale, 1 = no effect)")
ax.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
ax.minorticks_off()
ax.invert_yaxis()
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
save(fig, "drivers")

# 3. invite rules in the simulator
pol = pd.read_csv(tables / "sim_policies.csv")
# a dot chart, not bars: the differences are small and bars would have to start at 0
fig, ax = plt.subplots(figsize=(6.5, 3.8))
rules = pol.rule.unique()
for k, (scorer, color, marker) in enumerate([("MOVE C", ORANGE, "s"), ("learned", BLUE, "o")]):
    vals = pol.set_index(["scorer", "rule"]).loc[scorer, "value per plan"].reindex(rules)
    x = np.arange(len(rules)) + (k - 0.5) * 0.2
    ax.plot(x, vals, marker=marker, color=color, linestyle="none", markersize=8, label=f"{scorer} score")
    for xi, v in zip(x, vals):
        ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=INK)
ax.set(xticks=range(len(rules)), xticklabels=rules, ylabel="value per plan", xlim=(-0.5, len(rules) - 0.5))
ax.legend(loc="upper left")
save(fig, "invite_rules")

# 4. recovery speed by half-life
rec = pd.read_csv(tables / "sim_recovery.csv").query("target == 0.75")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for scorer, color, marker in [("MOVE C", ORANGE, "s"), ("learned", BLUE, "o")]:
    ax.plot(rec.half_life, rec[scorer], color=color, marker=marker, label=f"{scorer} score")
ax.set(xticks=rec.half_life, xlabel="half-life of past behaviour (days)", ylabel="reliable plans needed to reach 0.75",
       ylim=(0, None))
ax.legend(loc="upper left")
save(fig, "recovery")

# 5. matching vs greedy, by how many users are available
mat = pd.read_csv(tables / "sim_matching.csv")
fig, ax = plt.subplots(figsize=(5.5, 3.6))
for method, color, marker in [("greedy", ORANGE, "s"), ("integer programme", BLUE, "o")]:
    d = mat[mat.method == method]
    ax.plot(d["available users"], d["plans that happen"], color=color, marker=marker, label=method)
ax.set(xticks=mat["available users"].unique(), xlabel="users available per day (10 plans)", ylabel="share of plans that happen")
ax.legend(loc="lower right")
save(fig, "matching")

# what MOVE's labels mean: score C (rating part at its prior, as there are no ratings) on the no-show test period
df, _, y, *_ = noshow_data()
te = split(df)[1].index
preds = pd.read_parquet(root / "data/processed/preds_noshow_all.parquet")
score = np.floor(100 * (0.7 * preds["MOVE C"] + 0.3 * 0.625) + 0.5)
label = pd.cut(score, [-1, 49, 74, 100], labels=["Low show-up", "Building trust", "Trusted"])
labels = pd.DataFrame({"label": label.values, "showed": y[te].values, "bayes": preds["hierarchical Bayes, online"].values})
labels = labels.groupby("label", observed=True).agg(visits=("showed", "size"), observed_show_rate=("showed", "mean"),
                                                     bayes_predicted=("bayes", "mean"))
labels.to_csv(tables / "move_labels.csv")
print(labels.round(3).to_string())
