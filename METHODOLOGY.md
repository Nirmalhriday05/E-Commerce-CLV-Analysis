# Methodology Notes

Three decisions in this project are non-obvious. Each is recorded here with the evidence
behind it, so the reasoning is auditable rather than taken on trust.

---

## 1. Why `log1p` before clustering

RFM values are heavily right-skewed. Clustering on raw scaled values produces a technically
"better" silhouette score, but the clusters are useless:

| Transform | Silhouette | Calinski-Harabasz | Cluster sizes | Smallest |
|---|---|---|---|---|
| Raw + StandardScaler | **0.580** | 4,333 | 3,848 / 2,012 / 18 | 0.31% |
| log1p + StandardScaler | 0.348 | **5,217** | 2,397 / 2,268 / 1,213 | 20.64% |

The raw-scaled silhouette is inflated because K-Means isolates 18 extreme outliers into
their own cluster. A cluster containing 0.31% of customers is a list of anomalies, not a
marketing segment. Calinski-Harabasz — which does not reward tiny isolated clusters the
same way — prefers the log-transformed version.

**Decision:** log1p, because the goal is actionable segments. Stated plainly, this trades a
clustering-quality metric for business usability, and that trade should be visible rather
than hidden.

### Choice of k

| k | Inertia | Silhouette | Smallest cluster |
|---|---|---|---|
| 2 | 8,589 | 0.438 | 2,352 |
| **3** | **6,352** | **0.348** | **1,213** |
| 4 | 4,919 | 0.365 | 1,188 |
| 5 | 4,098 | 0.342 | 462 |

k=2 scores highest but collapses the distinction between active and high-value customers.
k=3 sits at the elbow and produces three segments that map onto distinct actions. k=4 is
statistically comparable; k=3 was kept for interpretability.

---

## 2. Segment names

Names follow the standard RFM segmentation vocabulary (Champions / Loyal Customers /
Hibernating) rather than invented labels, and are assigned by ranked median monetary value
rather than by hand — so re-running the pipeline cannot silently reassign them.

| Segment | Customers | % of base | % of revenue | Median recency | Median frequency |
|---|---|---|---|---|---|
| Champions | 1,213 | 20.6% | **74.7%** | 16 days | 13 orders |
| Loyal Customers | 2,268 | 38.6% | 19.9% | 58 days | 4 orders |
| Hibernating | 2,397 | 40.8% | 5.4% | 393 days | 1 order |

The Champions segment holding 20.6% of customers and 74.7% of revenue is an independent
corroboration of the headline concentration finding (top 20% of customers = 77.3% of
revenue), arrived at by clustering rather than by sorting.

---

## 3. Model: what it predicts, and where it does not beat a simple rule

### The problem with the original setup

The earlier model predicted `monetary` from `recency` and `frequency` on the same time
period. That is not a forecast — frequency and monetary are mechanically correlated within
a fixed window, so a high R² there measures a tautology, not predictive skill.

### Corrected setup

A temporal holdout, which is how customer-value models are properly validated:

- **Calibration window:** 2009-12-01 to 2011-06-08 (18 months) — features built here
- **Holdout window:** 2011-06-09 to 2011-12-09 (6 months) — target measured here
- **Target:** money actually spent in the following six months (0 if the customer did not return)
- 4,966 customers in calibration; 52.0% returned during the holdout

Nine features: recency, frequency, monetary, tenure, average order value, items, distinct
products, purchase rate, average gap between orders.

Two stages, because customer value is zero-inflated — most of the uncertainty is *whether*
someone returns, not how much they spend once they do:

1. **Will they return?** XGBoost classifier
2. **How much, given they return?** XGBoost regressor on returners only
3. **Expected value** = P(return) × E[spend | return]

### Results

| Metric | Value |
|---|---|
| Return-prediction AUC (holdout) | 0.793 |
| Return-prediction AUC (5-fold CV) | 0.776 ± 0.018 |
| Spearman correlation with actual future spend | 0.607 |
| Top-20% of ranked customers captures | 64.5% of future revenue |

### The honest part

**For ranking customers by future value, this model does not beat simply ranking by past
spend.**

| Approach | Top-20% revenue capture (holdout) | 5-fold CV |
|---|---|---|
| Two-stage XGBoost | 64.5% | 72.9% ± 5.2 |
| Rank by past spend | 65.2% | 72.6% ± 5.1 |
| Perfect foresight | 88.0% | — |

The difference is inside the noise. If the business question is *"who should we target?"*,
sorting by historical spend is as effective as gradient boosting, and is simpler to explain,
cheaper to run, and easier to trust.

Where the model does add something a sort cannot: a **calibrated probability that a given
customer returns at all** (AUC 0.776 ± 0.018, stable across folds). That supports decisions
a ranking cannot — retention budget per customer, expected-value thresholds, and separating
"high past spend, likely gone" from "high past spend, likely to return."

### A note on metric choice

Dollar-space R² was rejected as a headline metric. Across 5-fold CV it ranged from **-0.47
to 0.91** (std 0.52), because a single customer spending $184,016 dominates whichever fold
contains them. Any R² quoted from a single split of this data is an artifact of that split.
AUC and top-decile revenue capture are stable (std 0.018 and 5.2 respectively) and are what
is reported.

---

## Summary

The clustering choice trades a metric for usability. The segment names follow convention and
are assigned programmatically. The model is validated the way a forecast should be, and the
comparison against a trivial baseline is reported even though the baseline is competitive —
because a model that does not beat a sort is worth knowing about before it goes into
production, not after.
