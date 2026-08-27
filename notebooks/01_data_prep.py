"""
Corrected RFM data preparation for the CLV project.

THE BUG THIS FIXES
------------------
The original notebook normalised column names with:

    raw.columns = (raw.columns.str.strip().str.lower()
                   .str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))

That only inserts an underscore where a non-alphanumeric character already exists.
"Customer ID" has a space, so it correctly became "customer_id" -- but "InvoiceDate"
has no separator, so it became "invoicedate", NOT "invoice_date".

The next line kept only columns in a hard-coded list that asked for "invoice_date",
so the date column was silently dropped. The RFM aggregation then hit its fallback:

    recency = ("invoice_date", lambda s: ...) if "invoice_date" in raw.columns
              else ("customer_id", "size")

so `recency` was populated with the ROW COUNT per customer, not days since last
purchase. That is why recency ranged up to 12,890 on a 738-day dataset.

Frequency and monetary were never affected.

THE FIX
-------
Split camelCase before lowercasing, so "InvoiceDate" -> "invoice_date".
The assertion below makes the failure loud instead of silent if it ever recurs.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Paths are resolved relative to THIS FILE, not the current working directory,
# so the script behaves the same however it is launched.
HERE = Path(__file__).resolve().parent          # ...\CLV_project\Notebooks
PROJECT_ROOT = HERE.parent                       # ...\CLV_project
DATA_DIR = PROJECT_ROOT / "Data"
OUT_DIR = HERE / "Outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- load ----------
excel_path = DATA_DIR / "online_retail_II.xlsx"
xls = pd.ExcelFile(excel_path)
raw = pd.concat([xls.parse(s) for s in xls.sheet_names], ignore_index=True)
print("Loaded:", raw.shape)

# ---------- normalise columns (FIXED) ----------
raw.columns = (
    raw.columns.str.strip()
    .str.replace(r"(?<=[a-z0-9])(?=[A-Z])", "_", regex=True)  # camelCase -> snake_case
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)
print("Columns:", list(raw.columns))

# Fail loudly rather than silently falling back.
required = {"invoice", "invoice_date", "quantity", "price", "customer_id"}
missing = required - set(raw.columns)
assert not missing, f"Required columns missing after normalisation: {missing}"

# ---------- clean ----------
raw["invoice_date"] = pd.to_datetime(raw["invoice_date"], errors="coerce")
for c in ("quantity", "price"):
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

raw = raw.dropna(subset=["invoice", "invoice_date", "quantity", "price", "customer_id"])
raw = raw[(raw["quantity"] > 0) & (raw["price"] > 0)]
raw["amount"] = raw["quantity"] * raw["price"]
print("Cleaned:", raw.shape)
print("Date range:", raw["invoice_date"].min(), "->", raw["invoice_date"].max())

# ---------- RFM ----------
ref_date = raw["invoice_date"].max() + pd.Timedelta(days=1)
print("Reference date:", ref_date)

rfm = (
    raw.groupby("customer_id")
    .agg(
        recency=("invoice_date", lambda s: (ref_date - s.max()).days),
        frequency=("invoice", "nunique"),
        monetary=("amount", "sum"),
    )
    .reset_index()
)
rfm["customer_id"] = rfm["customer_id"].astype(int)

# Sanity check: recency cannot exceed the span of the dataset.
span = (raw["invoice_date"].max() - raw["invoice_date"].min()).days + 1
assert rfm["recency"].max() <= span, (
    f"Recency max {rfm['recency'].max()} exceeds dataset span {span} days -- "
    "date handling is broken."
)
print(f"Recency range: {rfm['recency'].min()}-{rfm['recency'].max()} days (span {span})")
print(f"Customers: {len(rfm):,} | Revenue: ${rfm['monetary'].sum():,.0f}")

# ---------- clustering ----------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

FEATURES = ["recency", "frequency", "monetary"]
# log1p first: raw RFM is heavily right-skewed, and clustering on raw values
# isolates a handful of outliers instead of producing usable segments.
X = StandardScaler().fit_transform(np.log1p(rfm[FEATURES].values))
rfm["cluster"] = KMeans(n_clusters=3, n_init=10, random_state=42).fit_predict(X)

profile = rfm.groupby("cluster")["monetary"].median().sort_values()
ordered = list(profile.index)
SEGMENTS = {
    ordered[0]: "Hibernating",
    ordered[1]: "Loyal Customers",
    ordered[2]: "Champions",
}
rfm["segment"] = rfm["cluster"].map(SEGMENTS)

rfm[["customer_id"] + FEATURES + ["cluster"]].to_csv(
    OUT_DIR / "rfm_with_clusters.csv", index=False
)
rfm[["customer_id"] + FEATURES + ["cluster", "segment"]].to_csv(
    OUT_DIR / "rfm_with_clusters_and_segments.csv", index=False
)

total = rfm["monetary"].sum()
summary = (
    rfm.groupby("cluster")
    .agg(
        customers=("customer_id", "size"),
        median_recency=("recency", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        total_revenue=("monetary", "sum"),
    )
    .reset_index()
)
summary["segment"] = summary["cluster"].map(SEGMENTS)
summary["pct_customers"] = (summary["customers"] / len(rfm) * 100).round(1)
summary["pct_revenue"] = (summary["total_revenue"] / total * 100).round(1)
summary.to_csv(OUT_DIR / "cluster_summary.csv", index=False)
print(summary[["segment", "customers", "pct_customers", "pct_revenue"]].to_string(index=False))

# ---------- marketing cohorts (explicit rules) ----------
med_mon = rfm["monetary"].median()
lapse = rfm["recency"].quantile(0.60)

cohorts = {
    "targets_vip": rfm.nlargest(int(round(len(rfm) * 0.20)), "monetary"),
    "targets_core": rfm[rfm["segment"] == "Champions"],
    "targets_atrisk": rfm[(rfm["monetary"] > med_mon) & (rfm["recency"] >= lapse)],
}
for name, df in cohorts.items():
    df[["customer_id"] + FEATURES].to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"{name:15s} n={len(df):5d}  revenue {df['monetary'].sum()/total*100:5.1f}%")

# ---------- CLV modelling datasets (temporal holdout) ----------
# Written here so clv_model.py never needs to re-read the Excel, and so the
# calibration/holdout split is defined in exactly one place.
CUTOFF = pd.Timestamp("2011-06-09")   # ~18 months calibration, ~6 months holdout

def build_features(df, ref):
    g = df.groupby("customer_id")
    f = pd.DataFrame({
        "recency": (ref - g.invoice_date.max()).dt.days,
        "frequency": g.invoice.nunique(),
        "monetary": g.amount.sum(),
        "tenure": (ref - g.invoice_date.min()).dt.days,
        "avg_order_value": g.amount.sum() / g.invoice.nunique(),
        "n_items": g.quantity.sum(),
        "n_products": g.stock_code.nunique(),
    }).reset_index()
    f["purchase_rate"] = f.frequency / f.tenure.clip(lower=1) * 30
    f["avg_gap"] = f.tenure / f.frequency.clip(lower=1)
    return f

calib = raw[raw.invoice_date <= CUTOFF]
hold = raw[raw.invoice_date > CUTOFF]

train = build_features(calib, CUTOFF + pd.Timedelta(days=1))
train = train.merge(
    hold.groupby("customer_id").amount.sum().rename("future_value"),
    on="customer_id", how="left",
).fillna({"future_value": 0.0})
train.to_csv(OUT_DIR / "clv_training_data.csv", index=False)

# Features for every customer at the end of the data, for scoring.
score = build_features(raw, raw.invoice_date.max() + pd.Timedelta(days=1))
score.to_csv(OUT_DIR / "clv_scoring_features.csv", index=False)

print(f"clv_training_data  n={len(train):,} | returned in holdout: {(train.future_value>0).mean():.1%}")
print(f"clv_scoring_features n={len(score):,}")

print("\nDone.")
