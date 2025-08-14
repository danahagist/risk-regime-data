# src/compute_macro_monthly.py
# -*- coding: utf-8 -*-

"""
Compute monthly macro dataset from FRED and write to data/macro_monthly_history.csv.

Rules:
- Dates truncated to first of month (YYYY-MM-01).
- Daily series -> monthly median.
- Monthly series -> monthly value.
- Quarterly series -> forward-filled to monthly at quarter starts.
- Compute MoM and YoY percentage changes where meaningful.
- Macro Core 3 signals:
    * INDPRO YoY > 0        -> indpro_signal = 1 else 0
    * USSLIND YoY > 0       -> lei_signal = 1 else 0
    * Yield curve > 0 bps   -> yield_curve_signal = 1 else 0
  macro_score = indpro_signal + lei_signal + yield_curve_signal
- SP500 is pulled from FRED (daily), aggregated to monthly median.
  sp500_earnings_yield is included as a placeholder column (NaN), since earnings are not from FRED.

Environment:
- Requires env var FRED_API_KEY.
"""

import os
import sys
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from fredapi import Fred

# -----------------------
# Config
# -----------------------

START_DATE = "1990-01-01"

# FRED series codes
SERIES = {
    # Core 3
    "INDPRO": "INDPRO",                # monthly
    "USSLIND": "USSLIND",              # monthly (Leading Index)
    "DGS10": "DGS10",                  # daily
    "DGS3MO": "DGS3MO",                # daily
    # Additional indicators
    "BAMLH0A0HYM2": "BAMLH0A0HYM2",    # daily (HY OAS)
    "UNRATE": "UNRATE",                # monthly
    "CPIAUCSL": "CPIAUCSL",            # monthly (CPI level, SA)
    "A191RL1Q225SBEA": "A191RL1Q225SBEA",  # quarterly, Real GDP QoQ SAAR (%)
    "UMCSENT": "UMCSENT",              # monthly
    "HOUST": "HOUST",                  # monthly
    "SP500": "SP500",                  # daily (S&P 500 index level)
}

OUTFILE = "data/macro_monthly_history.csv"


# -----------------------
# Helpers
# -----------------------

def get_fred() -> Fred:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("ERROR: FRED_API_KEY is not set in environment.", file=sys.stderr)
        sys.exit(1)
    return Fred(api_key=api_key)


def to_month_start_index(idx: pd.Index) -> pd.DatetimeIndex:
    """
    Convert any DatetimeIndex to first-of-month timestamps.
    """
    dt = pd.to_datetime(idx)
    return dt.to_period("M").to_timestamp(how="start")


def monthly_from_daily_median(s: pd.Series) -> pd.Series:
    """
    For daily series: take monthly median and stamp to month start.
    """
    s = s.dropna()
    if s.empty:
        return s
    # groupby month using PeriodIndex, then median, then to month-start timestamps
    grp = s.groupby(s.index.to_period("M")).median()
    grp.index = grp.index.to_timestamp(how="start")
    return grp.sort_index()


def monthly_from_monthly_value(s: pd.Series) -> pd.Series:
    """
    For monthly series: ensure index is month-start and sorted.
    """
    s = s.dropna()
    if s.empty:
        return s
    s.index = to_month_start_index(s.index)
    return s.sort_index()


def monthly_from_quarterly_ffill(s: pd.Series) -> pd.Series:
    """
    For quarterly series (e.g., Real GDP QoQ SAAR): convert to quarter start,
    then resample monthly and forward-fill within quarter.
    """
    s = s.dropna()
    if s.empty:
        return s
    # Coerce to period Q, then to timestamp at quarter start
    q = s.copy()
    q.index = pd.to_datetime(q.index).to_period("Q").to_timestamp(how="start")
    # Resample monthly and forward fill
    m = q.resample("MS").ffill()
    return m.sort_index()


def pct_change_safe(s: pd.Series, periods: int) -> pd.Series:
    """
    Safe percent change that returns NaN when not enough history.
    """
    return s.pct_change(periods=periods) * 100.0


def diff_bps(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    (a - b) * 100 to get basis points if a,b are in %.
    """
    return (a - b) * 100.0


def join_left(df: pd.DataFrame, name: str, s: pd.Series) -> pd.DataFrame:
    return df.join(s.rename(name), how="outer")


# -----------------------
# Fetch & Transform
# -----------------------

def fetch_series(fred: Fred, code: str, start: str) -> pd.Series:
    """
    Fetch series from FRED as a pandas Series indexed by Timestamp.
    """
    s = fred.get_series(code, observation_start=start)
    if s is None or len(s) == 0:
        return pd.Series(dtype=float)
    s = s.rename(code)
    s.index = pd.to_datetime(s.index)
    return s


def build_monthly_frame() -> pd.DataFrame:
    fred = get_fred()

    # Fetch raw
    raw = {}
    for k, code in SERIES.items():
        try:
            raw[k] = fetch_series(fred, code, START_DATE)
        except Exception as e:
            print(f"[WARN] Failed to fetch {code}: {e}", file=sys.stderr)
            raw[k] = pd.Series(dtype=float)

    # Aggregate to monthly
    indpro_m = monthly_from_monthly_value(raw["INDPRO"])
    lei_m    = monthly_from_monthly_value(raw["USSLIND"])
    dgs10_m  = monthly_from_daily_median(raw["DGS10"])
    dgs3m_m  = monthly_from_daily_median(raw["DGS3MO"])
    hy_m     = monthly_from_daily_median(raw["BAMLH0A0HYM2"])
    unrate_m = monthly_from_monthly_value(raw["UNRATE"])
    cpi_m    = monthly_from_monthly_value(raw["CPIAUCSL"])
    gdpq_m   = monthly_from_quarterly_ffill(raw["A191RL1Q225SBEA"])
    umc_m    = monthly_from_monthly_value(raw["UMCSENT"])
    houst_m  = monthly_from_monthly_value(raw["HOUST"])
    spx_m    = monthly_from_daily_median(raw["SP500"])

    # Start building output frame with a union of all month starts
    idx = None
    for s in [indpro_m, lei_m, dgs10_m, dgs3m_m, hy_m, unrate_m,
              cpi_m, gdpq_m, umc_m, houst_m, spx_m]:
        idx = s.index if idx is None else idx.union(s.index)

    if idx is None:
        return pd.DataFrame()

    idx = pd.DatetimeIndex(sorted(idx))
    df = pd.DataFrame(index=idx)

    # Join base columns
    df = join_left(df, "indpro", indpro_m)
    df = join_left(df, "lei", lei_m)
    df = join_left(df, "dgs10", dgs10_m)
    df = join_left(df, "dgs3mo", dgs3m_m)
    df = join_left(df, "hy_oas", hy_m)
    df = join_left(df, "unrate", unrate_m)
    df = join_left(df, "cpi", cpi_m)  # CPI level index (not YoY yet)
    df = join_left(df, "gdp_saar", gdpq_m)  # QoQ SAAR (%), quarterly ffilled monthly
    df = join_left(df, "umcsent", umc_m)
    df = join_left(df, "houst", houst_m)
    df = join_left(df, "sp500", spx_m)

    # -----------------------
    # Changes (MoM, YoY)
    # -----------------------
    # For level series, compute MoM and YoY % changes.
    # Note: For gdp_saar (already a rate), MoM and YoY aren’t conceptually perfect,
    # but included for uniformity (they’ll be diffs of the reported rate across months).
    def add_changes(prefix: str):
        s = df[prefix]
        df[f"{prefix}_mom"] = pct_change_safe(s, periods=1)
        df[f"{prefix}_yoy"] = pct_change_safe(s, periods=12)

    for col in ["indpro", "lei", "dgs10", "dgs3mo", "hy_oas",
                "unrate", "cpi", "gdp_saar", "umcsent", "houst", "sp500"]:
        add_changes(col)

    # -----------------------
    # Yield Curve (bps) + its changes
    # -----------------------
    df["yield_curve_spread"] = diff_bps(df["dgs10"], df["dgs3mo"])
    df["yield_curve_mom"] = pct_change_safe(df["yield_curve_spread"], periods=1)
    df["yield_curve_yoy"] = pct_change_safe(df["yield_curve_spread"], periods=12)

    # -----------------------
    # Core 3 signals & macro_score
    # -----------------------
    # INDPRO signal: YoY > 0
    df["indpro_signal"] = (df["indpro_yoy"] > 0).astype("Int64")

    # LEI signal: YoY > 0
    df["lei_signal"] = (df["lei_yoy"] > 0).astype("Int64")

    # Yield curve signal: spread > 0 bps
    df["yield_curve_signal"] = (df["yield_curve_spread"] > 0).astype("Int64")

    # Sum of the 3 signals (handle NA as 0 in the sum)
    sigs = df[["indpro_signal", "lei_signal", "yield_curve_signal"]].fillna(0).astype(int)
    df["macro_score"] = sigs.sum(axis=1)

    # Placeholder for earnings yield (we don’t have earnings via FRED)
    df["sp500_earnings_yield"] = np.nan

    # Clean up index to first-of-month and finalize
    df.index = to_month_start_index(df.index)
    df = df.sort_index()

    # Drop rows that are entirely NA across the main base fields (optional)
    base_cols = ["indpro", "lei", "dgs10", "dgs3mo", "hy_oas", "unrate", "cpi",
                 "gdp_saar", "umcsent", "houst", "sp500"]
    df = df.dropna(axis=0, how="all", subset=base_cols)

    # Reorder columns for readability
    ordered_cols = [
        # Core inputs
        "indpro", "indpro_mom", "indpro_yoy", "indpro_signal",
        "lei", "lei_mom", "lei_yoy", "lei_signal",
        "dgs10", "dgs10_mom", "dgs10_yoy",
        "dgs3mo", "dgs3mo_mom", "dgs3mo_yoy",
        "yield_curve_spread", "yield_curve_mom", "yield_curve_yoy", "yield_curve_signal",
        "macro_score",
        # Additional indicators
        "hy_oas", "hy_oas_mom", "hy_oas_yoy",
        "unrate", "unrate_mom", "unrate_yoy",
        "cpi", "cpi_mom", "cpi_yoy",
        "gdp_saar", "gdp_saar_mom", "gdp_saar_yoy",
        "umcsent", "umcsent_mom", "umcsent_yoy",
        "houst", "houst_mom", "houst_yoy",
        "sp500", "sp500_mom", "sp500_yoy",
        "sp500_earnings_yield",
    ]
    # Keep only those that exist (in case of missing series)
    ordered_cols = [c for c in ordered_cols if c in df.columns]
    df = df[ordered_cols]

    return df


# -----------------------
# Main
# -----------------------

def main():
    df = build_monthly_frame()
    if df.empty:
        print("No data fetched; nothing to write.", file=sys.stderr)
        sys.exit(2)

    # Ensure output folder exists
    outdir = os.path.dirname(OUTFILE)
    if outdir and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    # Write CSV with date column (YYYY-MM-01)
    out = df.copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(OUTFILE, index=False)
    print(f"Wrote {OUTFILE} with {len(out):,} rows and {len(out.columns):,} columns.")


if __name__ == "__main__":
    main()