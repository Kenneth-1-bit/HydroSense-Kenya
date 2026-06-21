import pandas as pd
import numpy as np
from config import WEATHER_LIMITS, SOIL_LIMITS


# ─────────────────────────────────────────────────────────────────────────────
# DATE HANDLING
# ─────────────────────────────────────────────────────────────────────────────

def fix_dates(df, column):

    df = df.copy()
    df[column] = pd.to_datetime(df[column], infer_datetime_format=True, errors="coerce")
    n_bad = df[column].isna().sum()
    if n_bad:
        print(f"  [fix_dates] Dropped {n_bad} row(s) with un-parseable '{column}' values.")
    return df[df[column].notna()].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DUPLICATES
# ─────────────────────────────────────────────────────────────────────────────

def drop_duplicates(df, subset):

    df = df.copy()
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [drop_duplicates] Removed {n_dropped} duplicate row(s) on {subset}.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC COERCION
# ─────────────────────────────────────────────────────────────────────────────

def convert_numeric(df, columns):

    df = df.copy()
    for col in columns:
        before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col], errors="coerce")
        after = df[col].isna().sum()
        if after > before:
            print(f"  [convert_numeric] '{col}': {after - before} non-numeric value(s) → NaN.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# RAINFALL FILL
# ─────────────────────────────────────────────────────────────────────────────

def fill_rainfall(df):

    df = df.copy()

    valid_count = df["rainfall_mm"].notna().sum()
    if valid_count == 0:
        raise ValueError(
            "Cannot fill rainfall: all values are missing. "
            "Check the raw data source."
        )

    mean_rain = df["rainfall_mm"].mean()
    n_missing = df["rainfall_mm"].isna().sum()

    if n_missing:
        df["rainfall_mm"] = df["rainfall_mm"].fillna(mean_rain)
        print(f"  [fill_rainfall] {n_missing} missing value(s) filled with mean = {mean_rain:.4f} mm.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# RANGE FILTERING
# ─────────────────────────────────────────────────────────────────────────────

def range_filter(df, limits):

    df = df.copy()
    for col, (lo, hi) in limits.items():
        if col not in df.columns:
            continue
        mask = (df[col] < lo) | (df[col] > hi)
        n_out = mask.sum()
        if n_out:
            print(f"  [range_filter] '{col}': {n_out} row(s) outside [{lo}, {hi}] dropped.")
            df = df[~mask]
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# IQR OUTLIER DETECTION AND REMOVAL
# ─────────────────────────────────────────────────────────────────────────────

def iqr_bounds(series):

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def remove_outliers(df, columns):

    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        if df[col].isna().any():
            print(f"  [remove_outliers] Warning: '{col}' still contains NaN. "
                  "Fill missing values before running IQR removal.")
        lo, hi = iqr_bounds(df[col].dropna())
        mask   = (df[col] < lo) | (df[col] > hi)
        n_out  = mask.sum()
        if n_out:
            print(f"  [remove_outliers] '{col}': {n_out} row(s) outside "
                  f"IQR fence [{lo:.3f}, {hi:.3f}] dropped.")
            df = df[~mask]
    return df.reset_index(drop=True)
