# src/clv_model.py
"""
Customer value model: predicts spend over the next six months.

WHAT CHANGED AND WHY
--------------------
The previous version of this script trained on

    X = df[["recency", "frequency", "cluster"]]
    y = df["monetary"]

using a single time period. Two problems:

1. `cluster` was produced by K-Means fitted on monetary, so it leaked the target
   directly into the features.
2. Predicting monetary from recency/frequency measured within the same window is
   not a forecast. Those quantities are mechanically correlated inside a fixed
   window, so a high R2 there reflects a tautology rather than predictive skill.

This version uses a temporal holdout: features are built from an 18-month
calibration window, and the target is money actually spent in the following six
months. Customer value is zero-inflated (about half of customers do not return),
so it is modelled in two stages:

    expected value = P(returns) x E[spend | returns]

Metric choice matters here. Dollar-space R2 on this dataset swings between -0.47
and 0.91 across cross-validation folds, because a single customer spending
$184,016 dominates whichever fold contains them. AUC and top-decile revenue
capture are stable, so those are what this script reports.

Run 01_data_prep.py first; it writes the two input files used below.
"""

from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier, XGBRegressor

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Notebooks" / "Outputs"

FILE_TRAIN = OUT / "clv_training_data.csv"
FILE_SCORE = OUT / "clv_scoring_features.csv"
FILE_PRED_OUT = OUT / "predicted_customer_value.csv"
FILE_PRED_DETAIL = OUT / "clv_predictions_detail.csv"
METRICS_PATH = OUT / "model_metrics.json"
MODEL_PATH = OUT / "model_xgb.pkl"

FEATURES = [
    "recency", "frequency", "monetary", "tenure", "avg_order_value",
    "n_items", "n_products", "purchase_rate", "avg_gap",
]
SEED = 42

for p in (FILE_TRAIN, FILE_SCORE):
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run 01_data_prep.py first.")

# ---------- Load ----------
train = pd.read_csv(FILE_TRAIN)
X = train[FEATURES].values
y = train["future_value"].values
y_bin = (y > 0).astype(int)

print(f"Customers: {len(train):,} | returned during holdout: {y_bin.mean():.1%}")

# ---------- Validate on a held-out slice ----------
tr, te = train_test_split(
    np.arange(len(train)), test_size=0.2, random_state=SEED, stratify=y_bin
)

clf = XGBClassifier(random_state=SEED, verbosity=0, eval_metric="logloss")
clf.fit(X[tr], y_bin[tr])
proba = clf.predict_proba(X[te])[:, 1]

returners = tr[y_bin[tr] == 1]
reg = XGBRegressor(random_state=SEED, verbosity=0)
reg.fit(X[returners], np.log1p(y[returners]))

expected = np.expm1(reg.predict(X[te])).clip(0)
predicted_value = proba * expected

auc = roc_auc_score(y_bin[te], proba)
pr_auc = average_precision_score(y_bin[te], proba)
rho = float(spearmanr(predicted_value, y[te]).statistic)

n20 = int(len(te) * 0.20)
capture = y[te][np.argsort(-predicted_value)[:n20]].sum() / y[te].sum()
baseline = y[te][np.argsort(-train["monetary"].values[te])[:n20]].sum() / y[te].sum()

# Cross-validated AUC, because a single split is not evidence of stability.
cv_aucs = []
for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y_bin):
    m = XGBClassifier(random_state=SEED, verbosity=0, eval_metric="logloss")
    m.fit(X[a], y_bin[a])
    cv_aucs.append(roc_auc_score(y_bin[b], m.predict_proba(X[b])[:, 1]))

print("\nModel evaluation (temporal holdout)")
print(f"  Return AUC          : {auc:.4f}   (5-fold CV {np.mean(cv_aucs):.4f} +/- {np.std(cv_aucs):.4f})")
print(f"  Return PR-AUC       : {pr_auc:.4f}")
print(f"  Spearman vs actual  : {rho:.4f}")
print(f"  Top-20% capture     : {capture:.1%}")
print(f"  Baseline (past spend): {baseline:.1%}   <- model must beat this to be worth using")

if capture <= baseline:
    print("\n  NOTE: for ranking customers, sorting by past spend is as good as this model.")
    print("        The model's distinct contribution is the calibrated return probability.")

# ---------- Refit on all calibration data, then score everyone ----------
clf_full = XGBClassifier(random_state=SEED, verbosity=0, eval_metric="logloss")
clf_full.fit(X, y_bin)

reg_full = XGBRegressor(random_state=SEED, verbosity=0)
reg_full.fit(X[y_bin == 1], np.log1p(y[y_bin == 1]))

score = pd.read_csv(FILE_SCORE)
Xs = score[FEATURES].values
score["return_probability"] = clf_full.predict_proba(Xs)[:, 1]
score["expected_spend"] = np.expm1(reg_full.predict(Xs)).clip(0)
score["predicted_value"] = (score["return_probability"] * score["expected_spend"]).round(2)

OUT.mkdir(parents=True, exist_ok=True)
score[["customer_id", "predicted_value"]].to_csv(FILE_PRED_OUT, index=False)
score[["customer_id", "return_probability", "expected_spend", "predicted_value"]].round(4).to_csv(
    FILE_PRED_DETAIL, index=False
)
print(f"\nPredictions saved to: {FILE_PRED_OUT}  ({len(score):,} customers)")

metrics = {
    "model": "xgboost_two_stage",
    "validation": "temporal holdout (18mo calibration -> 6mo holdout)",
    "return_auc": round(float(auc), 4),
    "return_auc_cv_mean": round(float(np.mean(cv_aucs)), 4),
    "return_auc_cv_std": round(float(np.std(cv_aucs)), 4),
    "return_pr_auc": round(float(pr_auc), 4),
    "spearman_vs_actual_future_value": round(rho, 4),
    "top20pct_revenue_capture": round(float(capture), 4),
    "baseline_top20pct_revenue_capture": round(float(baseline), 4),
    "n_train": int(len(tr)),
    "n_test": int(len(te)),
    "features": FEATURES,
    "target": "spend in next 6 months",
}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to: {METRICS_PATH}")

with open(MODEL_PATH, "wb") as f:
    pickle.dump({"classifier": clf_full, "regressor": reg_full, "features": FEATURES}, f)
print(f"Model saved to: {MODEL_PATH}")
