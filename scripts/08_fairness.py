from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from thesis.data.datasets import noshow_data
from thesis.data.features import split
from thesis.eval.metrics import evaluate

root = Path(__file__).resolve().parent.parent
tables = root / "results/tables"
MODELS = ["MOVE C", "logistic regression", "LightGBM + isotonic", "hierarchical Bayes, online"]
d = noshow_data()
df, X = d["df"], pd.get_dummies(d["X"], drop_first=True, dtype=float)
train, test = (part.index for part in split(df))
t = df.loc[test].reset_index(drop=True)
showed = 1 - t.no_show.values
preds = pd.read_parquet(root / "data/processed/preds_noshow_all.parquet").reset_index(drop=True)
# groups fixed in advance. Area deprivation: neighbourhoods in thirds by their share of welfare recipients in training
welfare_share = df.loc[train].groupby("neighbourhood").scholarship.mean()
area = pd.qcut(welfare_share, 3, labels=["low", "middle", "high"])
groups = {"age": pd.cut(t.age, [-1, 17, 39, 64, 120], labels=["0-17", "18-39", "40-64", "65+"]).astype(str),
          "sex": t.gender, "welfare": t.scholarship.map({0: "no", 1: "yes"}), "disability": t.handicap.map({0: "no", 1: "yes"}),
          "chronic condition": ((t.hypertension + t.diabetes) > 0).map({False: "no", True: "yes"}),
          "area welfare share": t.neighbourhood.map(area).astype(str).replace("nan", "unknown")}


def riskiest(risk, share=0.2, seed=0):
    # flag exactly `share` of visits, ties broken at random (MOVE C gives many patients the same score)
    order = np.lexsort((np.random.default_rng(seed).random(len(risk)), -risk))
    flagged = np.zeros(len(risk), dtype=bool)
    flagged[order[:int(share * len(risk))]] = True
    return flagged


def gaps(rows):
    # predicted minus observed show rate, per group and model
    out = {}
    for g, labels in groups.items():
        lab = labels.values[rows]
        for level in np.unique(lab):
            m = lab == level
            for model in MODELS:
                out[(g, level, model)] = preds[model].values[rows][m].mean() - showed[rows][m].mean()
    return out


point = gaps(np.arange(len(t)))
rng = np.random.default_rng(0)
rows_of = t.groupby("patient_id").indices
patients = np.array(list(rows_of))
boot = pd.DataFrame([gaps(np.concatenate([rows_of[p] for p in rng.choice(patients, len(patients))])) for _ in range(200)])
rows = []
for (g, level, model), gap in point.items():
    m = (groups[g] == level).values
    flagged = riskiest(1 - preds[model].values)
    metrics = evaluate(showed[m], preds[model].values[m])
    rows.append({"group": g, "level": level, "model": model, "visits": m.sum(), "observed show": showed[m].mean(),
                 "calibration gap": gap, "gap low": boot[(g, level, model)].quantile(0.025),
                 "gap high": boot[(g, level, model)].quantile(0.975), "cal_slope": metrics["cal_slope"], "auc": metrics["auc"],
                 "flagged": flagged[m].mean(), "no-shows caught": flagged[m & (showed == 0)].sum() / (m & (showed == 0)).sum()})
out = pd.DataFrame(rows)
out.to_csv(tables / "fairness_subgroups.csv", index=False)
# does leaving welfare status out of the model change how welfare recipients are treated?
rows = []
for name, cols in [("with welfare status", X.columns), ("without welfare status", X.columns.drop("scholarship"))]:
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(X.loc[train, cols], 1 - df.no_show[train])
    p = lr.predict_proba(X.loc[test, cols])[:, 1]
    flagged = riskiest(1 - p)
    for level, m in [("welfare yes", t.scholarship.values == 1), ("welfare no", t.scholarship.values == 0)]:
        rows.append({"model": name, "group": level, "observed show": showed[m].mean(), "predicted show": p[m].mean(),
                     "flagged": flagged[m].mean(), "log loss (all)": evaluate(showed, p)["log_loss"]})
unaware = pd.DataFrame(rows)
unaware.to_csv(tables / "fairness_without_welfare.csv", index=False)
show = out[out.model == "hierarchical Bayes, online"].drop(columns="model")
print(show.round(3).to_string(index=False))
print(unaware.round(4).to_string(index=False))
