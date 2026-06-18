"""
features.py — date features, lag features, rolling stats, promo flags.
Called by train.py and inference.py.
"""
import pandas as pd
import numpy as np


MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

LAG_DAYS      = [7, 14, 28]          # sales lags
ROLLING_WINS  = [7, 14, 28]          # rolling stats windows

FEATURE_COLS = None   # set after build; used by inference


def _promo2_active(row):
    """Return 1 if Promo2 is active for this store on this date."""
    if row["Promo2"] == 0 or row["PromoInterval"] == "None":
        return 0
    month = row["Date"].month
    intervals = [MONTH_MAP[m] for m in row["PromoInterval"].split(",") if m in MONTH_MAP]
    return int(month in intervals)


def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["Store", "Date"], inplace=True)

    # --- Date features ---
    df["Year"]        = df["Date"].dt.year
    df["Month"]       = df["Date"].dt.month
    df["Week"]        = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfMonth"]  = df["Date"].dt.day
    df["DayOfYear"]   = df["Date"].dt.dayofyear
    # Cyclical encoding (sine/cosine) so model understands month 12 ≈ month 1
    df["Month_sin"]   = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]   = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"]     = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"]     = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # Days to / from nearest state holiday (simple flag for now)
    df["IsStateHoliday"]  = (df["StateHoliday"] != 0).astype(int)

    # --- Promo2 active flag ---
    if "PromoInterval" in df.columns:
        df["Promo2Active"] = df.apply(_promo2_active, axis=1)
    else:
        df["Promo2Active"] = 0

    # --- Lag features (per store) ---
    if is_train:
        for lag in LAG_DAYS:
            df[f"Sales_lag_{lag}"] = (
                df.groupby("Store")["Sales"].shift(lag)
            )

        # --- Rolling statistics ---
        for win in ROLLING_WINS:
            df[f"Sales_roll_mean_{win}"] = (
                df.groupby("Store")["Sales"]
                  .shift(1)                        # avoid leakage: shift before rolling
                  .rolling(win, min_periods=1)
                  .mean()
                  .values
            )
            df[f"Sales_roll_std_{win}"] = (
                df.groupby("Store")["Sales"]
                  .shift(1)
                  .rolling(win, min_periods=1)
                  .std()
                  .fillna(0)
                  .values
            )

        # Drop rows where lag features are NaN (first N rows per store)
        max_lag = max(LAG_DAYS)
        df = df.dropna(subset=[f"Sales_lag_{max_lag}"]).copy()

    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    drop_cols = {
        "Date", "Sales", "Customers", "Open",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    }
    return [c for c in df.columns if c not in drop_cols]
