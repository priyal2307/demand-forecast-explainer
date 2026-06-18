# Demand Forecast Explainer

**Retail sales forecasting with SHAP-powered explanations** — built on the Rossmann Store Sales dataset.

A production-style forecasting system for retail demand, combining LightGBM gradient boosting with quantile regression (confidence bands) and SHAP explainability, served via a Streamlit dashboard.

---

## What this is

A store manager or analyst selects a retail store. The system:

1. **Forecasts** daily sales — median prediction + 80% confidence band (Q10–Q90)
2. **Explains** the forecast — top SHAP drivers translated to plain English ("Promotion active: +€320 impact")
3. **Simulates what-ifs** — toggle promotions on/off, see forecast change live
4. **Scores in batch** — upload a CSV of stores, get all forecasts at once

---

## Quick start

```bash
git clone <repo-url>
cd demand-forecast-explainer

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build all artifacts from scratch
python3 src/clean.py
PYTHONPATH=src python3 src/train.py

# Launch dashboard
python3 -m streamlit run app/copilot.py
# Opens at http://localhost:8501
```

Prebuilt artifacts ship in `artifacts/` — skip to the last step to demo immediately.

---

## Architecture

```
data/raw/train.csv + store.csv
        │
        ▼
src/clean.py  ──►  data/processed/train_ready.csv
        │
        └──► src/train.py
                ├── artifacts/model_median.pkl   (LightGBM regression)
                ├── artifacts/model_q10.pkl      (quantile 10%)
                ├── artifacts/model_q90.pkl      (quantile 90%)
                ├── artifacts/feature_list.json
                └── artifacts/metrics.json

         At inference time:
         src/inference.py  Forecaster.predict(X)
           1. Load saved models
           2. Predict median + Q10/Q90
           3. SHAP TreeExplainer → per-prediction drivers
           4. Plain-English driver mapping
                       │
                       ▼
              app/copilot.py (Streamlit)
              ├── Store Forecast page (time-series + SHAP cards + weekly pattern)
              ├── Model Performance page (metrics, feature importance, distributions)
              └── Batch Upload page (CSV scoring + download)
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.9+ | ML + Streamlit standard |
| Data | pandas 2.x | Tabular manipulation |
| Modeling | LightGBM 4.x | Strong on tabular, handles categoricals natively |
| Quantile regression | LightGBM `objective="quantile"` | Honest forecast bands instead of point estimates |
| Explainability | SHAP TreeExplainer | Per-prediction feature contributions |
| UI | Streamlit | Fast path to a polished demo |
| Plots | Plotly | Interactive charts |

---

## Key modeling decisions

| Decision | What | Why |
|---|---|---|
| **Time-based split** | Train < May 2015, val May–Jun, test > Jun | Shuffling time-series causes leakage and inflates metrics |
| **Lag features** | Sales lagged 7/14/28 days | LightGBM can't model temporal order on its own |
| **Rolling stats** | 7/14/28-day rolling mean + std, shifted by 1 | Captures recent trend without leaking future data |
| **Quantile regression** | Q10 + Q90 alongside median | Honest uncertainty — "sales will be €4k–€6k" not a false point estimate |
| **Cyclical encoding** | sin/cos for month, day-of-week | Month 12 and month 1 are adjacent; raw integers don't encode that |
| **Metric: MAPE + naive baseline** | Always compare to same-weekday-last-week | Raw MAPE alone is meaningless without context |
| **Drop closed days** | Rows where Open=0 are filtered | Forecasting €0 on known-closed days inflates accuracy |

---

## Results

| Metric | Value |
|---|---|
| Test MAPE | **9.4%** |
| Naive baseline MAPE (same weekday last week) | 36.2% |
| Improvement over baseline | **74% better** |
| Test RMSE | €901 |
| Training rows | 728,628 |
| Features | 30 |

---

## Dataset

**Rossmann Store Sales** (Kaggle) — daily sales for 1,115 German drug stores, 2013–2015.
- 1,017,209 raw rows · 133,235 used for training (open-store days only)
- Features: store type, assortment, competition distance, promotions, holidays, macro date context

---

## Limitations

- **Data staleness** — trained on 2013–2015 data; would need retraining on current sales data for production use
- **No external features** — weather, economic indicators, or nearby events could improve accuracy
- **Lag features require history** — for truly new stores with no sales history, lag/rolling features fall back to zeros
- **Point-in-time only** — the current UI forecasts based on the last known date in training; extending to future dates requires a rolling inference approach

---

## Next steps

- Add future-date inference (forecast the next 7/14/28 days forward, not just in-sample)
- Optuna hyperparameter sweep (50 trials)
- Per-store model or store-cluster segmentation
- Monthly retraining pipeline with drift detection
- Weather API integration as an external feature
