# Customer Lifetime Value (CLV) Predictor 📊

**Author:** Nirmmal Hriday N R
**Contact:** nirmalhriday12804@gmail.com
**GitHub:** [Nirmalhriday05](https://github.com/Nirmalhriday05)
**Institution:** Persevex Data Science Internship

---

## 🎯 Project Overview

A dual-application platform for analyzing and predicting customer lifetime value, built on RFM (Recency, Frequency, Monetary) analysis and a two-stage machine learning model. Built with Python, Dash, and XGBoost.

### Two Integrated Applications:
1. **CLV / RFM Dashboard** (`src/app.py`, port 3000): K-Means clustering, RFM segmentation, and predicted-value browsing
2. **Universal Predictor** (`universal_predictor/app.py`, port 3100): Upload any transaction dataset and get a freshly trained CLV model — dataset-agnostic by design (see note below)

---

## 📊 Dataset Information

**Source:** Online Retail II, UK-based e-commerce transactions
**Link:** https://www.kaggle.com/datasets/alpaypasali/online-retail-ii

- **Customers:** 5,878 unique customers
- **Total revenue:** $17,743,429
- **Time period:** December 1, 2009 – December 9, 2011 (739 days)

---

## 🐛 The Bug That Shaped This Project

The original column-normalization step only inserted underscores where a separator already existed. `"InvoiceDate"` has no separator, so it silently became `"invoicedate"` instead of `"invoice_date"` — and the date column was dropped without any error. The RFM pipeline's fallback then quietly filled `recency` with each customer's **row count** instead of days since last purchase, producing impossible values (max recency of 12,890 "days" on a 739-day dataset).

**Fix:** a camelCase-aware column normalizer, plus assertions that fail loudly instead of falling back silently (`01_data_prep.py`, `verify_data.py`).

This also meant the original K-Means clustering — run on corrupted recency values — was invalid and had to be redone from scratch.

---

## 📈 Customer Segmentation (K-Means, k=3, corrected)

| Segment | Customers | % of Base | % of Revenue | Median Recency | Median Frequency |
|---|---|---|---|---|---|
| **Champions** | 1,213 | 20.6% | **74.7%** | 16 days | 13 orders |
| Loyal Customers | 2,268 | 38.6% | 19.9% | 58 days | 4 orders |
| Hibernating | 2,397 | 40.8% | 5.4% | 393 days | 1 order |

**Revenue concentration check (independent method — sorting, not clustering):** the top 20% of customers by spend generate **77.3%** of total revenue ($13.71M). This closely matches the Champions cluster (20.6% of customers → 74.7% of revenue) found via a completely different method — two independent approaches agreeing is stronger evidence than either alone.

Full reasoning for the `log1p` transform and choice of k=3 is in [`METHODOLOGY.md`](METHODOLOGY.md).

---

## 🔬 Predictive Model: Two-Stage XGBoost

The original single-period model predicted `monetary` from `recency`/`frequency`/`cluster` in the same time window — but `cluster` was derived from `monetary`, so it leaked the target directly, and predicting within one window isn't a real forecast at all.

**Corrected approach — temporal holdout:**
- Calibration window: 18 months of history → Holdout window: the following 6 months
- **Stage 1:** XGBoost classifier — will this customer return? (`return_probability`)
- **Stage 2:** XGBoost regressor, trained only on returners — how much will they spend? (`expected_spend`)
- **Final score:** `predicted_value = return_probability × expected_spend`

### Results

| Metric | Value |
|---|---|
| Return AUC (holdout) | 0.797 |
| Return AUC (5-fold CV) | 0.778 ± 0.017 |
| Top-20% revenue capture | 64.2% |
| Baseline (rank by past spend) | 65.2% |

**Honest disclosure:** for *ranking* customers, this model does **not** beat the simple baseline of sorting by past spend — the difference is within noise. Its real value is the **calibrated return probability**, which a simple sort can't give you. This is disclosed openly rather than hidden, and is discussed further in `METHODOLOGY.md`.

Dollar-space R² was rejected as a metric — a single $184K customer swings it from -0.47 to 0.91 across CV folds. AUC and top-decile capture are stable and are what's reported.

---

## 📥 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the data pipeline (from src/)
python 01_data_prep.py

# 3. Train the model
python clv_model.py

# 4. Verify outputs
python verify_data.py

# 5. Launch the dashboard (from src/)
python app.py
# Dashboard: http://127.0.0.1:3000

# 6. In a separate terminal, launch the Universal Predictor (from universal_predictor/)
python app.py
# Universal Predictor: http://127.0.0.1:3100
# Dashboard:         http://127.0.0.1:3000
# Universal Predictor: http://127.0.0.1:3100
```

---

## 📁 Repository Structure

```
├── src/
│   ├── app.py              # CLV/RFM Dashboard
│   ├── clv_model.py         # Two-stage XGBoost model
│   └── verify_data.py       # Sanity checks on pipeline outputs
├── universal_predictor/
│   └── app.py                # Dataset-agnostic predictor tool
├── notebooks/
│   ├── 01_data_prep.py / .ipynb
│   ├── 02_model_training_xgboost.ipynb
│   └── Outputs/               # Generated CSVs, model, metrics
├── METHODOLOGY.md             # Detailed reasoning behind every modeling decision
└── README.md
```

---

## 🛠️ Technologies Used

- Python 3.8+, Pandas, NumPy, Scikit-learn
- XGBoost (two-stage model)
- Dash & Plotly, Dash Bootstrap Components

---

## 💡 A Note on the Universal Predictor

The Universal Predictor doesn't load a saved model — every upload triggers a fresh `train_test_split` + fit. This is a deliberate trade-off: it keeps the tool dataset-agnostic at the cost of per-run training time, which is fine for exploratory use but wouldn't suit high-throughput production.

---

## 👨‍💻 Author

**Nirmmal Hriday N R**
📧 nirmalhriday12804@gmail.com
