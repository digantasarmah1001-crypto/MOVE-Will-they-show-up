from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
tables, out = root / "results/tables", root / "results/latex"
out.mkdir(parents=True, exist_ok=True)
NAMES = {"MOVE C": "Hand-made score", "tuned MOVE C": "Hand-made score, learned prior", "base rate": "Average rate",
         "logistic regression": "Logistic regression", "LightGBM + isotonic": "LightGBM, isotonic",
         "hierarchical Bayes": "Hierarchical Bayes", "hierarchical Bayes, online": "Hierarchical Bayes, updated",
         "fixed rate": "Fixed rule", "no overbooking": "No overbooking", "perfect foresight": "Perfect foresight"}


def tex(text):
    text = str(text)
    for a, b in [("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]:
        text = text.replace(a, b)
    return text


def cell(x, digits):
    if isinstance(x, (float, np.floating)):
        return "" if np.isnan(x) else f"{x:,.{digits}f}"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return tex(NAMES.get(x, x))


def save(df, name, digits=3, align=None):
    align = align or "l" + "r" * (len(df.columns) - 1)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(tex(c) for c in df.columns) + " \\\\", "\\midrule"]
    lines += [" & ".join(cell(x, digits) for x in row) + " \\\\" for row in df.itertuples(index=False)]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / f"{name}.tex").write_text("\n".join(lines) + "\n")


def interval(low, high, scale=1, digits=3):
    return [f"{l * scale:.{digits}f} to {h * scale:.{digits}f}" for l, h in zip(low, high)]


read = lambda name: pd.read_csv(tables / f"{name}.csv")
c = read("cleaning_log")
save(c[["dataset", "step", "rows_removed", "rows_left"]].rename(columns={"rows_removed": "Removed", "rows_left": "Left"})
     .replace({"noshow": "No-show", "hotel": "Hotel"}).rename(columns=str.capitalize), "cleaning")
d = pd.concat([read("hier_diagnostics_noshow").assign(Model="No-show"), read("hier_diagnostics_hotel").assign(Model="Hotel")])
save(d[["Model", "divergences", "max_r_hat", "min_ess_bulk"]].rename(
    columns={"divergences": "Divergences", "max_r_hat": "Largest $\\hat R$", "min_ess_bulk": "Smallest ESS"}), "diagnostics")
for name in ["noshow", "hotel"]:
    r = read(f"rq2_comparison_{name}")
    a = r[r.stratum == "all"]
    save(pd.DataFrame({"Model": a.model, "Log loss": a.log_loss, "Brier": a.brier, "AUC": a.auc, "Cal. slope": a.cal_slope,
                       "Cal. intercept": a.cal_intercept, "vs hand-made (95% CI)": a.vs_MOVE_ci.where(a.model != "MOVE C", "")}),
         f"prediction_{name}")
r = read("rq2_comparison_noshow")
save(r.pivot_table(index="model", columns="stratum", values="log_loss", sort=False)[["0", "1-2", "3+", "all"]]
     .reset_index().rename(columns={"model": "Model", "0": "No earlier visit", "1-2": "1 to 2", "3+": "3 or more", "all": "All"}),
     "prediction_new_patients")
save(read("rq1_drivers").fillna("").rename(columns=str.capitalize), "drivers", align="lrrc")
s = read("sms_outage_check")
save(pd.DataFrame({"Booked 3+ days ahead": s.eligible.map({0: "no", 1: "yes"}), "Outage day": s.outage.map({0: "no", 1: "yes"}),
                   "Visits": s.visits, "No-show rate": s.no_show, "Got an SMS": s.sms}), "sms_outage", align="llrrr")
e = read("sms_effect")
save(pd.DataFrame({"Estimate": e.estimate, "Value": e.value, "95% CI": interval(e.low, e.high), "E-value": e.e_value}),
     "sms_effect", align="lrrr")
b = read("sms_effect_by_risk")
save(pd.DataFrame({"Risk third": b["risk group"], "Visits": b["eligible visits"], "No-show without SMS": b["no-show without SMS"],
                   "Absolute effect (95% CI)": [f"{v:.3f} ({l:.3f} to {h:.3f})" for v, l, h in zip(b["absolute effect"], b["absolute low"], b["absolute high"])],
                   "Risk ratio (95% CI)": [f"{v:.2f} ({l:.2f} to {h:.2f})" for v, l, h in zip(b["risk ratio"], b["ratio low"], b["ratio high"])]}),
     "sms_by_risk")
f = read("fairness_subgroups").query("model == 'hierarchical Bayes, online'")
save(pd.DataFrame({"Group": f.group, "Level": f.level, "Visits": f.visits, "Observed show": f["observed show"],
                   "Calibration gap (95% CI)": [f"{g:.3f} ({l:.3f} to {h:.3f})" for g, l, h in zip(f["calibration gap"], f["gap low"], f["gap high"])],
                   "Slope": f.cal_slope, "Flagged": f.flagged, "No-shows caught": f["no-shows caught"]}), "fairness", align="llrrrrrr")
w = read("fairness_without_welfare")
save(w.rename(columns=str.capitalize), "fairness_welfare", digits=4)
dc = read("decision_curves")
dc = dc[dc.threshold.isin([0.1, 0.2, 0.3]) & dc.version.fillna("").isin(["", "level updated"])]
save(dc.pivot_table(index="strategy", columns="threshold", values="net benefit", sort=False).reset_index()
     .rename(columns={"strategy": "Strategy", 0.1: "Threshold 0.1", 0.2: "Threshold 0.2", 0.3: "Threshold 0.3"}), "net_benefit", digits=4)
dd = read("decision_curve_differences")
save(pd.DataFrame({"Threshold": dd.threshold, "Against": dd.against, "Difference": dd["median"], "95% CI": interval(dd.low, dd.high, digits=4)}),
     "net_benefit_differences", digits=4, align="rlrr")
p = read("reminders_prevented")
save(p[p["share reminded"] < 0.5].pivot_table(index="model", columns="share reminded", values="measured per risk third", sort=False)
     .reset_index().rename(columns={"model": "Ranking", 0.1: "Riskiest 10%", 0.2: "Riskiest 20%", 0.3: "Riskiest 30%"}),
     "prevented", digits=1)
k = read("clinic_policies")
save(pd.DataFrame({"Overflow cost": k["overflow cost"], "Policy": k.policy, "Cost per day": k.cost, "Extra booked": k.extra_booked,
                   "Empty slots": k.idle_slots, "Overflow": k.overflow, "Days with overflow": k.clinic_days_with_overflow}),
     "clinic", digits=2, align="rlrrrrr")
g = read("clinic_differences")
save(pd.DataFrame({"Overflow cost": g["overflow cost"], "Against": g.against, "Difference": g.difference,
                   "95% CI": interval(g.low, g.high, digits=2)}), "clinic_differences", digits=2, align="rlrr")
o = read("robustness_hotel")
save(o.pivot_table(index="driver", columns="version", values="odds ratio", sort=False).reset_index().rename(columns={"driver": "Driver"}),
     "robustness", digits=2)
bal = pd.read_csv(tables / "sms_balance.csv").rename(columns={"Unnamed: 0": "Variable"})
save(bal.rename(columns=str.capitalize), "balance")
save(read("score_ties").rename(columns=str.capitalize), "ties")
print(sorted(p.name for p in out.glob("*.tex")))
