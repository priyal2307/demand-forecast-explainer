"""
train.py — LightGBM regressor (median + quantile) with time-based split.
Saves: artifacts/model_median.pkl, model_q10.pkl, model_q90.pkl,
       artifacts/feature_list.json, artifacts/metrics.json
"""
import sys, json, pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features, get_feature_cols

PROC      = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

PARAMS_BASE = dict(
    boosting_type   = "gbdt",
    n_estimators    = 1000,
    learning_rate   = 0.05,
    num_leaves      = 127,
    min_child_samples = 20,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 0.1,
    random_state    = 42,
    n_jobs          = -1,
    verbose         = -1,
)


def naive_baseline(y_true, y_naive):
    """MAPE of a naive same-weekday-last-week baseline."""
    mask = y_naive > 0
    return mean_absolute_percentage_error(y_true[mask], y_naive[mask])


def train():
    print("Loading processed data…")
    df = pd.read_csv(PROC / "train_ready.csv", low_memory=False)
    df = build_features(df, is_train=True)

    feature_cols = get_feature_cols(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # --- Time-based split (no shuffle!) ---
    df["Date"] = pd.to_datetime(df["Date"])
    cutoff_val  = pd.Timestamp("2015-05-01")
    cutoff_test = pd.Timestamp("2015-06-15")

    train_df = df[df["Date"] <  cutoff_val]
    val_df   = df[(df["Date"] >= cutoff_val) & (df["Date"] < cutoff_test)]
    test_df  = df[df["Date"] >= cutoff_test]

    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    X_train, y_train = train_df[feature_cols], train_df["Sales"]
    X_val,   y_val   = val_df[feature_cols],   val_df["Sales"]
    X_test,  y_test  = test_df[feature_cols],  test_df["Sales"]

    # --- Naive baseline: same weekday, previous week ---
    naive_sales = test_df["Sales_lag_7"]
    mask = naive_sales > 0
    naive_mape  = mean_absolute_percentage_error(y_test[mask], naive_sales[mask])
    print(f"Naive baseline MAPE: {naive_mape:.4f}")

    # --- Train median model ---
    print("Training median model…")
    model_median = lgb.LGBMRegressor(objective="regression", **PARAMS_BASE)
    model_median.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)]
    )

    # --- Train quantile models ---
    print("Training q10 model…")
    model_q10 = lgb.LGBMRegressor(objective="quantile", alpha=0.10, **PARAMS_BASE)
    model_q10.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])

    print("Training q90 model…")
    model_q90 = lgb.LGBMRegressor(objective="quantile", alpha=0.90, **PARAMS_BASE)
    model_q90.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)])

    # --- Evaluate ---
    preds = model_median.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mape  = mean_absolute_percentage_error(y_test, preds)
    print(f"Test RMSE: {rmse:.1f}   MAPE: {mape:.4f}   Naive MAPE: {naive_mape:.4f}")

    # --- Save artifacts ---
    for name, obj in [("model_median", model_median),
                      ("model_q10", model_q10),
                      ("model_q90", model_q90)]:
        with open(ARTIFACTS / f"{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    with open(ARTIFACTS / "feature_list.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    metrics = dict(
        test_rmse         = round(rmse, 2),
        test_mape         = round(mape, 4),
        naive_mape        = round(naive_mape, 4),
        n_train           = len(train_df),
        n_val             = len(val_df),
        n_test            = len(test_df),
        cutoff_val        = str(cutoff_val.date()),
        cutoff_test       = str(cutoff_test.date()),
        n_features        = len(feature_cols),
    )
    with open(ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("All artifacts saved ✓")
    return metrics


if __name__ == "__main__":
    train()
