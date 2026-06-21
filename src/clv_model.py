# src/clv_model.py
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from lightgbm import LGBMRegressor

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parents[1]                      # ...\CLV_project
OUT  = ROOT / "Notebooks" / "Outputs"                           # ...\CLV_project\Notebooks\Outputs

FILE_RFM      = OUT / "rfm_with_clusters_and_segments.csv"
FILE_PRED_OUT = OUT / "predicted_customer_value.csv"
METRICS_PATH  = OUT / "model_metrics.json"
MODEL_PATH    = OUT / "model_lgbm.pkl"

# ---------- Load & prepare ----------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(FILE_RFM)

    # keep only rows with the core RFM fields
    df = df.dropna(subset=["monetary", "recency", "frequency"]).copy()

    # ensure numeric cluster; fall back to 0 if missing
    df["cluster"] = pd.to_numeric(df.get("cluster", 0), errors="coerce").fillna(0).astype(int)

    # make sure customer_id is a string (for later merge in app)
    if "customer_id" in df.columns:
        df["customer_id"] = df["customer_id"].astype(str)
    else:
        # if not present, create an index-based id so the file is still valid
        df["customer_id"] = df.index.astype(str)

    return df

df = load_data()

X = df[["recency", "frequency", "cluster"]]
y = df["monetary"].astype(float)

# log-transform target to handle skew; we'll invert after prediction
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# ---------- Train ----------
model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------- Evaluate (invert log back to monetary scale) ----------
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)         # back to original $
y_test_real = np.expm1(y_test)

rmse = float(np.sqrt(mean_squared_error(y_test_real, y_pred)))
r2   = float(r2_score(y_test_real, y_pred))

print("✅ Model Evaluation Results:")
print(f"   RMSE: {rmse:,.2f}")
print(f"   R²  : {r2:.3f}")

# ---------- Predict whole dataset & save ----------
df_out = pd.DataFrame({
    "customer_id": df["customer_id"].astype(str),
    "predicted_value": np.expm1(model.predict(X))
})

# write predictions CSV
FILE_PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
df_out.to_csv(FILE_PRED_OUT, index=False)
print(f"\n💾 Predictions saved to: {FILE_PRED_OUT}")

# write metrics JSON
metrics = {"model": "lightgbm_log1p", "rmse": rmse, "r2": r2}
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"💾 Metrics saved to: {METRICS_PATH}")

# save the trained model for reuse
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
print(f"💾 Model saved to: {MODEL_PATH}")