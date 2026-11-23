from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from thesis.data.datasets import hotel_data, noshow_data
from thesis.data.features import split

root = Path(__file__).resolve().parent.parent
tables, figs = root / "results/tables", root / "results/figures"
figs.mkdir(parents=True, exist_ok=True)
# colour-blind safe palette, each series keeps its colour in every figure
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2,
                     "lines.markersize": 6, "legend.frameon": False, "font.size": 10})
data = {"noshow": noshow_data(), "hotel": hotel_data()}
test = {name: split(d["df"], d["date"], date)[1].index for (name, d), date in zip(data.items(), ["2016-05-25", "2016-12-31"])}
preds = {name: pd.read_parquet(root / f"data/processed/preds_{name}_all.parquet") for name in data}
titles = {"noshow": "no-show data", "hotel": "hotel data"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(figs / f"{name}.png", dpi=200)
    plt.close(fig)


# broken commitments by lead time
fig, ax = plt.subplots(figsize=(6, 3.8))
drivers = pd.read_csv(tables / "drivers.csv").query("driver == 'lead'")
for name, color, marker in [("noshow", BLUE, "o"), ("hotel", ORANGE, "s")]:
    d = drivers[drivers.dataset == name]
    ax.plot(d.level, d.rate, color=color, marker=marker, label=titles[name])
ax.set(xlabel="lead time (days)", ylabel="share of broken commitments", ylim=(0, None))
ax.legend(loc="upper left")
save(fig, "lead_time")
# reliability diagrams on the test period
models = {"hierarchical Bayes": (BLUE, "o"), "MOVE C": (ORANGE, "s"), "logistic regression": (AQUA, "^"),
          "LightGBM": (YELLOW, "D")}
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
for ax, name in zip(axes, data):
    y = data[name]["y"][test[name]]
    for model, (color, marker) in models.items():
        col = "hierarchical Bayes, online" if model == "hierarchical Bayes" and name == "noshow" else model
        observed, predicted = calibration_curve(y, preds[name][col], n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, color=color, marker=marker, label=model.replace("MOVE C", "hand-made score"))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--", label="perfect calibration")
    ax.set(title=titles[name], xlabel="predicted chance of showing up", ylabel="observed share who showed up",
           xlim=(0.3, 1), ylim=(0.3, 1))
axes[0].legend(loc="upper left")
save(fig, "reliability")
# drivers as odds ratios with 90% intervals
effects = [pd.read_csv(tables / "rq1_noshow_effects.csv", index_col=0), pd.read_csv(tables / "hier_effects_hotel.csv", index_col=0)]
drivers = [("same-day commitment", "same_day", None), ("lead time (per 1 sd)", "log_lead", "log_lead"),
           ("past broken commitments", "log_prior_no_shows", "log_prev_cancellations"),
           ("past kept commitments", "log_prior_shows", "log_prev_kept"), ("SMS reminder", "sms_received", None),
           ("group size", None, "group_size"), ("price per night", None, "adr")]
fig, ax = plt.subplots(figsize=(7, 4.4))
for k, (label, color, marker, shift) in enumerate([("no-show data", BLUE, "o", -0.12), ("hotel data", ORANGE, "s", 0.12)]):
    xs, ys = [], []
    for row, driver in enumerate(drivers):
        if driver[k + 1]:
            r = effects[k].loc[f"b[{driver[k + 1]}]"]
            ax.plot(np.exp([r.eti90_lb, r.eti90_ub]), [row + shift] * 2, color=color)
            xs.append(np.exp(r["mean"]))
            ys.append(row + shift)
    ax.plot(xs, ys, marker=marker, color=color, linestyle="none", label=label)
ax.axvline(1, color=MUTED, linewidth=1, linestyle="--")
ax.set(xscale="log", yticks=range(len(drivers)), yticklabels=[d[0] for d in drivers],
       xlabel="odds ratio of breaking the commitment (log scale, 1 = no effect)")
ax.set_xticks([0.2, 0.5, 1, 2], ["0.2", "0.5", "1", "2"])
ax.minorticks_off()
ax.invert_yaxis()
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
save(fig, "drivers")
# decision curves for reminders, on the later test days with the level updated
dc = pd.read_csv(tables / "decision_curves.csv")
fig, ax = plt.subplots(figsize=(6.5, 4))
for strategy, color, style in [("remind everyone", MUTED, "--"), ("MOVE C", ORANGE, "-"), ("logistic regression", AQUA, "-"),
                               ("hierarchical Bayes, online", BLUE, "-")]:
    d = dc[(dc.strategy == strategy) & (dc.version.fillna("").isin(["", "level updated"]))]
    ax.plot(d.threshold, d["net benefit"], color=color, linestyle=style, label=strategy.replace(", online", "").replace("MOVE C", "hand-made score"))
ax.axhline(0, color=INK, linewidth=1, label="remind no one")
ax.set(xlabel="risk threshold for a reminder", ylabel="net benefit (no-shows reached per patient)", ylim=(-0.02, 0.2))
ax.legend(loc="upper right")
save(fig, "decision_curves")
# and one table: what MOVE's labels mean in real show rates (rating part at its prior, as there are no ratings)
p = preds["noshow"]
score = np.floor(100 * (0.7 * p["MOVE C"] + 0.3 * 0.625) + 0.5)
labels = pd.DataFrame({"label": pd.cut(score, [-1, 49, 74, 100], labels=["Low show-up", "Building trust", "Trusted"]).values,
                       "showed": data["noshow"]["y"][test["noshow"]].values, "bayes": p["hierarchical Bayes, online"].values})
labels = labels.groupby("label", observed=True).agg(visits=("showed", "size"), observed_show_rate=("showed", "mean"),
                                                     bayes_predicted=("bayes", "mean"))
labels.to_csv(tables / "move_labels.csv")
print(labels.round(3).to_string())
