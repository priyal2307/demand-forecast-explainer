"""
inference.py — Forecaster class used by the Streamlit app and CLI.
Loads trained models, runs predictions, generates SHAP explanations.
"""
import json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap

ROOT      = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"

# Plain-English feature descriptions for SHAP driver cards
FEATURE_DESCRIPTIONS = {
    "Promo":              "Active promotion running today",
    "Promo2Active":       "Recurring Promo2 active this month",
    "SchoolHoliday":      "School holiday",
    "IsStateHoliday":     "State / public holiday",
    "DayOfWeek":          "Day of the week",
    "Month":              "Month of year",
    "Month_sin":          "Seasonal position (month cycle)",
    "Month_cos":          "Seasonal position (month cycle)",
    "DOW_sin":            "Day-of-week cycle position",
    "DOW_cos":            "Day-of-week cycle position",
    "Sales_lag_7":        "Sales same weekday last week",
    "Sales_lag_14":       "Sales same weekday 2 weeks ago",
    "Sales_lag_28":       "Sales same weekday 4 weeks ago",
    "Sales_roll_mean_7":  "7-day rolling average sales",
    "Sales_roll_mean_14": "14-day rolling average sales",
    "Sales_roll_mean_28": "28-day rolling average sales",
    "Sales_roll_std_7":   "Sales volatility (7-day)",
    "Sales_roll_std_14":  "Sales volatility (14-day)",
    "Sales_roll_std_28":  "Sales volatility (28-day)",
    "CompetitionDistance":"Distance to nearest competitor (m)",
    "CompetitionAge":     "How long competitor has been open",
    "StoreType":          "Store format/type",
    "Assortment":         "Product assortment level",
    "Promo2":             "Store enrolled in Promo2 programme",
    "Year":               "Calendar year",
    "Week":               "Week of year",
    "DayOfMonth":         "Day of month",
    "DayOfYear":          "Day of year",
}


class Forecaster:
    def __init__(self):
        self.model_median = self._load("model_median.pkl")
        self.model_q10    = self._load("model_q10.pkl")
        self.model_q90    = self._load("model_q90.pkl")
        with open(ARTIFACTS / "feature_list.json") as f:
            self.feature_cols = json.load(f)
        with open(ARTIFACTS / "metrics.json") as f:
            self.metrics = json.load(f)
        self._explainer = None   # lazy-loaded

    def _load(self, name):
        with open(ARTIFACTS / name, "rb") as f:
            return pickle.load(f)

    @property
    def explainer(self):
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.model_median)
        return self._explainer

    def predict(self, X: pd.DataFrame):
        """
        X: DataFrame with feature columns (one or many rows).
        Returns dict with median, q10, q90 arrays.
        """
        X = X[self.feature_cols]
        return {
            "median": self.model_median.predict(X),
            "q10":    self.model_q10.predict(X),
            "q90":    self.model_q90.predict(X),
        }

    def explain(self, X: pd.DataFrame, max_features: int = 5):
        """
        Returns list of dicts with top SHAP drivers for each row.
        Each dict: [{feature, description, shap_value, direction}]
        """
        X = X[self.feature_cols].reset_index(drop=True)
        shap_values = self.explainer.shap_values(X)

        results = []
        for i in range(len(X)):
            sv = shap_values[i]
            top_idx = np.argsort(np.abs(sv))[::-1][:max_features]
            drivers = []
            for idx in top_idx:
                feat = self.feature_cols[idx]
                val  = sv[idx]
                drivers.append({
                    "feature":     feat,
                    "description": FEATURE_DESCRIPTIONS.get(feat, feat),
                    "shap_value":  round(float(val), 1),
                    "direction":   "↑ increases forecast" if val > 0 else "↓ decreases forecast",
                    "raw_value":   round(float(X.iloc[i][feat]), 3),
                })
            results.append(drivers)
        return results

    def explain_single(self, row: pd.Series, max_features: int = 5):
        return self.explain(row.to_frame().T, max_features=max_features)[0]
