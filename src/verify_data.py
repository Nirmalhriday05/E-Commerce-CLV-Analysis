"""
Sanity checks for the CLV pipeline outputs.

Reads the specific files it needs rather than globbing for whichever CSV turns up
first, and asserts the invariants that would have caught the recency defect --
the earlier version reported statistics without ever asking whether they were
possible.
"""

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "Notebooks" / "Outputs"

FILE_RFM = OUTPUTS / "rfm_with_clusters_and_segments.csv"
FILE_PRED = OUTPUTS / "predicted_customer_value.csv"
FILE_METRICS = OUTPUTS / "model_metrics.json"

# Online Retail II covers 2009-12-01 to 2011-12-09.
DATASET_SPAN_DAYS = 739

problems = []

# ---------- RFM ----------
if not FILE_RFM.exists():
    print(f"MISSING: {FILE_RFM}")
    sys.exit(1)

rfm = pd.read_csv(FILE_RFM)
print(f"Dataset: {FILE_RFM.name}")
print("\n=== BASIC STATS ===")
print(f"Rows              : {len(rfm):,}")
print(f"Unique customers  : {rfm['customer_id'].nunique():,}")
print(f"Total revenue     : ${rfm['monetary'].sum():,.0f}")
print(f"Median monetary   : ${rfm['monetary'].median():,.2f}")
print(f"Median frequency  : {rfm['frequency'].median():,.1f}")
print(f"Recency range     : {rfm['recency'].min()}-{rfm['recency'].max()} days")

# ---------- Invariants ----------
print("\n=== SANITY CHECKS ===")


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail and not ok else ''}")
    if not ok:
        problems.append(label)


check(
    f"recency within dataset span ({DATASET_SPAN_DAYS}d)",
    rfm["recency"].max() <= DATASET_SPAN_DAYS,
    f"max is {rfm['recency'].max()} -- date handling is broken",
)
check("recency positive", rfm["recency"].min() >= 1)
check("frequency positive", rfm["frequency"].min() >= 1)
check("monetary positive", rfm["monetary"].min() > 0)
check("customer_id unique", rfm["customer_id"].is_unique)
check("no null segments", rfm["segment"].notna().all())

# Revenue concentration -- the project's headline claim.
top20 = rfm.nlargest(int(round(len(rfm) * 0.20)), "monetary")["monetary"].sum()
share = top20 / rfm["monetary"].sum()
print(f"\n  Top 20% of customers hold {share:.1%} of revenue")

# ---------- Segments ----------
print("\n=== SEGMENTS ===")
seg = rfm.groupby("segment").agg(
    customers=("customer_id", "size"), revenue=("monetary", "sum")
)
seg["pct_customers"] = (seg.customers / len(rfm) * 100).round(1)
seg["pct_revenue"] = (seg.revenue / rfm["monetary"].sum() * 100).round(1)
print(seg[["customers", "pct_customers", "pct_revenue"]].to_string())

# ---------- Predictions ----------
print("\n=== PREDICTIONS ===")
if FILE_PRED.exists():
    pred = pd.read_csv(FILE_PRED)
    print(f"Customers scored  : {pred['predicted_value'].notna().sum():,}")
    print(f"Mean predicted    : ${pred['predicted_value'].mean():,.2f}")
    print(f"Median predicted  : ${pred['predicted_value'].median():,.2f}")
    check("predictions non-negative", pred["predicted_value"].min() >= 0)
else:
    print("predicted_customer_value.csv not found")

# ---------- Metrics ----------
print("\n=== MODEL METRICS ===")
if FILE_METRICS.exists():
    m = json.load(open(FILE_METRICS, encoding="utf-8"))
    print(f"Model      : {m.get('model', 'Unknown')}")
    print(f"Validation : {m.get('validation', 'n/a')}")
    print(f"Return AUC : {m.get('return_auc', 'n/a')} "
          f"(CV {m.get('return_auc_cv_mean', 'n/a')} +/- {m.get('return_auc_cv_std', 'n/a')})")
    cap = m.get("top20pct_revenue_capture")
    base = m.get("baseline_top20pct_revenue_capture")
    if cap is not None and base is not None:
        print(f"Top-20% capture: {cap:.1%}  |  past-spend baseline: {base:.1%}")
        if cap <= base:
            print("  NOTE: model does not beat the baseline at ranking; "
                  "its value is the calibrated return probability.")
else:
    print("model_metrics.json not found")

# ---------- Result ----------
print()
if problems:
    print(f"{len(problems)} CHECK(S) FAILED: {', '.join(problems)}")
    sys.exit(1)
print("All checks passed.")
