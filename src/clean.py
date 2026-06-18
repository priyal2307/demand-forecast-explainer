"""
clean.py — merge train + store, handle missing values, filter closed days.
Output: data/processed/train_ready.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load_and_merge():
    train = pd.read_csv(RAW / "train.csv", dtype={"StateHoliday": str}, low_memory=False)
    store = pd.read_csv(RAW / "store.csv")

    # Merge store metadata
    df = train.merge(store, on="Store", how="left")

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])

    # Drop closed-store rows (Sales=0 when Open=0 — not useful to forecast)
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # --- Missing value strategy ---
    # CompetitionDistance: fill with large number (no nearby competition)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].max() * 2)

    # CompetitionOpenSince: fill 0 months/years → treat as very old competition
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0).astype(int)
    df["CompetitionOpenSinceYear"]  = df["CompetitionOpenSinceYear"].fillna(0).astype(int)

    # Promo2 info: fill with 0 / "None"
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
    df["PromoInterval"]   = df["PromoInterval"].fillna("None")

    # StateHoliday: normalise mixed 0 / '0'
    df["StateHoliday"] = df["StateHoliday"].replace("0", "None").fillna("None")

    # Encode categoricals
    df["StoreType"]   = df["StoreType"].astype("category").cat.codes
    df["Assortment"]  = df["Assortment"].astype("category").cat.codes
    df["StateHoliday"] = df["StateHoliday"].astype("category").cat.codes

    # Competition age in months (how long has competitor been open)
    df["CompetitionAge"] = (
        (df["Date"].dt.year  - df["CompetitionOpenSinceYear"])  * 12 +
        (df["Date"].dt.month - df["CompetitionOpenSinceMonth"])
    ).clip(lower=0)

    df.sort_values(["Store", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    out = PROC / "train_ready.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows → {out}")
    return df


if __name__ == "__main__":
    load_and_merge()
