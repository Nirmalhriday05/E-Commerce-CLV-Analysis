from pathlib import Path
import pandas as pd
import json

# === PATHS ===
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "Notebooks" / "Outputs"

# Automatically detect the correct CSV file
csv_candidates = list(OUTPUTS.glob("*.csv"))
if not csv_candidates:
    raise FileNotFoundError(f"No CSV files found in {OUTPUTS}")
else:
    # Prefer files that contain 'predicted' or 'rfm' in their names
    for f in csv_candidates:
        if "predicted" in f.name.lower() or "rfm" in f.name.lower():
            FILE_RFM = f
            break
    else:
        FILE_RFM = csv_candidates[0]

FILE_METRICS = OUTPUTS / "model_metrics.json"

print(f"\n✅ Using dataset: {FILE_RFM.name}")

# === LOAD DATA ===
rfm = pd.read_csv(FILE_RFM)

# === BASIC STATS ===
print("\n=== BASIC STATS CHECK ===")
print(f"Rows: {len(rfm):,}")
if "customer_id" in rfm.columns:
    print(f"Unique Customers: {rfm['customer_id'].nunique():,}")

if "monetary" in rfm.columns:
    print(f"Total Revenue: {rfm['monetary'].sum():,.0f}")
    print(f"Average Monetary: {rfm['monetary'].mean():,.2f}")
    print(f"Median Monetary: {rfm['monetary'].median():,.2f}")
if "frequency" in rfm.columns:
    print(f"Average Frequency: {rfm['frequency'].mean():,.2f}")
if "recency" in rfm.columns:
    print(f"Median Recency: {rfm['recency'].median():,.2f}")

# === PREDICTED VALUES ===
pred_cols = [c for c in rfm.columns if "predicted" in c.lower()]
if pred_cols:
    col = pred_cols[0]
    print(f"\nPredicted CLV Available: ✅ ({rfm[col].notna().sum():,} entries)")
    print(f"Avg Predicted CLV: {rfm[col].mean():,.2f}")
else:
    print("\nPredicted CLV Available: ❌ Not found")

# === MODEL METRICS ===
if FILE_METRICS.exists():
    with open(FILE_METRICS, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print("\n=== MODEL METRICS ===")
    print(f"Model: {metrics.get('model', 'Unknown')}")
    print(f"R²: {metrics.get('r2', 'N/A')}")
    print(f"RMSE: {metrics.get('rmse', 'N/A')}")
else:
    print("\nNo model_metrics.json found.")