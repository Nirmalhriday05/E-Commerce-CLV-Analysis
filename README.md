# Customer Lifetime Value (CLV) & Customer Segmentation 📊

**Author:** Nirmmal Hriday N R
**Contact:** nirmalhriday12804@gmail.com

---

## 🎯 Project Overview

An end-to-end customer analytics project on e-commerce transaction data. The goal was to answer two business questions:

1. **Who are our most valuable customers, and how concentrated is our revenue?**
2. **Can we predict a customer's value from their purchasing behaviour alone?**

The project covers the full pipeline — data cleaning, feature engineering, unsupervised segmentation, predictive modelling, and an interactive dashboard for non-technical stakeholders.

---

## 📦 Dataset

| | |
|---|---|
| Raw transaction records | ~1,000,000 |
| Unique customers after aggregation | **5,878** |
| Total revenue in scope | **$17,743,429** |

Transactions were cleaned (missing values, duplicates, returns) and aggregated to one row per customer.

---

## 🔧 Methodology

### 1. RFM Feature Engineering
Each customer was reduced to three behavioural features:

- **Recency** — days since their most recent purchase
- **Frequency** — number of purchases made
- **Monetary** — total amount spent

### 2. Feature Scaling
Recency is measured in days (tens), monetary in dollars (thousands). Without scaling, monetary would dominate any distance calculation purely because of its magnitude. `StandardScaler` was applied so all three features contribute proportionally.

### 3. K-Means Clustering
`k` was not assumed. Values from **k=2 to k=8** were evaluated using the **elbow method** and **silhouette score**, and **k=3** was selected as the best fit.

---

## 💡 Key Findings

### Revenue is highly concentrated

| Customer group | Share of revenue |
|---|---|
| **Top 20%** | **77.3%** ($13.71M) |
| Top 25% | 81.6% ($14.48M) |
| Bottom 40% | 3.9% |

A textbook Pareto distribution — roughly **a fifth of customers generate over three quarters of revenue**, which makes a strong case for concentrating retention spend rather than spreading acquisition budget evenly.

### Clustering surfaced a hidden account type

The clusters did **not** split into even value tiers. Instead:

| Cluster | Customers | % of base | % of revenue | Median spend | Median purchases |
|---|---|---|---|---|---|
| 0 — General base | 5,520 | 93.9% | 49.1% | $800 | 3 |
| 1 — Loyal / Active Core | 348 | 5.9% | 37.7% | $11,534 | 27 |
| 2 — High-volume outliers | **10** | 0.2% | **13.2%** | **$156,627** | **198** |

The most useful insight came from Cluster 2: **10 customers — 0.2% of the base — account for 13% of all revenue**, with median spend of ~$157K across ~198 orders each. That purchasing pattern doesn't describe a retail shopper; these are almost certainly **wholesale or B2B accounts** embedded in what was assumed to be a consumer dataset. They warrant separate treatment in both modelling and account management.

---

## 🤖 Predictive Modelling

**Objective:** predict customer monetary value from RFM features.
**Setup:** 80/20 train-test split, `log1p` transform on the target (spending is heavily right-skewed), metrics reported back in dollar space.

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **XGBoost** ✅ | **0.4257** | **$14,525** | $1,843 |
| LightGBM | 0.2856 | $16,201 | $1,839 |
| Linear Regression | **-66.91** | $157,953 | $8,373 |

**XGBoost was selected** as the final model.

### Why Linear Regression fails so badly here

The large negative R² is not a bug — it's the informative result. Because the target was log-transformed to handle skew, small errors in log space become **enormous** once predictions are converted back to dollars. Linear Regression cannot compensate for this, so its dollar-space predictions diverge wildly. Tree-based models are far more robust to it. This is the concrete justification for using gradient boosting rather than a linear baseline.

### Interpreting R² = 0.43

The model explains roughly 43% of the variance in customer value using only recency, frequency and monetary history. The remainder depends on factors these three features cannot capture (product mix, seasonality, marketing exposure). For a segmentation and prioritisation use case — reliably separating high-value from low-value customers — this is a usable signal, not a precision forecasting tool.

---

## 🖥️ Applications

### 1. CLV / RFM Dashboard (`src/`)
Interactive Dash/Plotly dashboard with 3D RFM scatter plots, segment and cluster filters, and KPI cards. Lets non-technical stakeholders explore segments without writing code.

> ⚠️ **Known issue:** the dashboard's "Model Type" field displays `UNKNOWN`. Investigation showed the dashboard is currently running LightGBM (its displayed R² of 0.286 matches the LightGBM benchmark above), while the model label was never wired through correctly. Fix in progress.

### 2. Universal Predictor (`universal_predictor/`)
A general-purpose tool rather than a single saved model. Upload any tabular dataset, select the ID / target / feature columns, and the app **trains a fresh model on that data** (XGBoost, LightGBM or Linear Regression) and returns predictions, metrics, feature importance and a downloadable report.

**Design note:** it deliberately retrains per upload instead of loading a pickled model. A saved model only generalises to data shaped exactly like its training set; retraining keeps the tool dataset-agnostic — hence "Universal." The trade-off is training latency per run, which is acceptable for an exploratory tool but would not suit a high-throughput production service.

---

## 🛠️ Tech Stack

- **Data:** Python, Pandas, NumPy
- **Modelling:** scikit-learn (KMeans, StandardScaler, LinearRegression), XGBoost, LightGBM
- **Visualisation:** Dash, Plotly
- **Environment:** Jupyter Notebooks

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── 01_data_prep.ipynb              # Cleaning, RFM features, scaling, K-Means
│   └── 02_model_training_xgboost.ipynb # Model training and evaluation
├── src/                                # CLV / RFM dashboard
├── universal_predictor/                # Reusable predictor application
└── requirements.txt
```

---

## 📝 Note on Tooling

AI coding assistants were used for parts of the implementation. The analytical decisions — feature selection, choice of `k`, model selection and evaluation, and interpretation of results — are my own, and all figures above were re-verified directly against the source data.
