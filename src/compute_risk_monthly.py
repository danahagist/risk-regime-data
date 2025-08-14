# src/compute_risk_monthly.py
# Build monthly risk dataset with levels, MoM, YoY, and regime signals.
# Sources: yfinance (^GSPC, ^VIX, ^VIX3M, HYG, LQD)

import os
import sys
import math
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

OUT_PATH = "data/risk_monthly_history.csv"

# -----------------------
# Helpers
# -----------------------

def get_close(series_df: pd.DataFrame) -> pd.Series:
    """Return an appropriate close/adj close series with a DateTimeIndex."""
    if "Adj Close" in series_df.columns:
        s = series_df["Adj Close"].copy()
    elif "Close" in series_df.columns:
        s = series_df["Close"].copy()
    else:
        raise ValueError("Downloaded data has no Close/Adj Close column.")
    s = s.dropna()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s

def to_month_start_from_daily(s: pd.Series) -> pd.Series:
    """
    Convert daily series to monthly (use month-end value) and set the index
    to the first day of each month (month-start) as requested.
    """
    s = s.sort_index()
    # Take the last obs each month (month end)
    s = s.resample("ME").last()
    # Convert to period M then to timestamp at month START
    s.index = s.index.to_period("M").to_timestamp(how="start")
    return s

def pct_mom(s: pd.Series) -> pd.Series:
    return (s / s.shift(1) - 1.0) * 100.0

def pct_yoy(s: pd.Series) -> pd.Series:
    return (s / s.shift(12) - 1.0) * 100.0

def last_nonnull_join(df_list, how="outer"):
    out = None
    for df in df_list:
        out = df if out is None else out.join(df, how=how)
    return out

# -----------------------
# Main
# -----------------------

def main():
    # Download daily history for all needed tickers
    tickers = ["^GSPC", "^VIX", "^VIX3M", "HYG", "LQD"]
    dl = yf.download(tickers, period="max", interval="1d", progress=False, auto_adjust=False, threads=True)

    # yfinance returns multi-index columns for multiple tickers
    series = {}
    for t in tickers:
        df_t = dl.xs(t, level=1, axis=1) if isinstance(dl.columns, pd.MultiIndex) else dl
        series[t] = get_close(df_t).rename(t)

    # Convert to monthly (index at month start)
    spx_m   = to_month_start_from_daily(series["^GSPC"]).rename("sp500")
    vix_m   = to_month_start_from_daily(series["^VIX"]).rename("vix")
    vix3m_m = to_month_start_from_daily(series["^VIX3M"]).rename("vix3m")
    hyg_m   = to_month_start_from_daily(series["HYG"]).rename("hyg")
    lqd_m   = to_month_start_from_daily(series["LQD"]).rename("lqd")

    # Core levels
    hyg_lqd_ratio = (hyg_m / lqd_m).rename("hyg_lqd_ratio")

    # MoM / YoY for displayed metrics
    sp500_mom = pct_mom(spx_m).rename("sp500_mom")
    sp500_yoy = pct_yoy(spx_m).rename("sp500_yoy")

    hyg_lqd_mom = pct_mom(hyg_lqd_ratio).rename("hyg_lqd_ratio_mom")
    hyg_lqd_yoy = pct_yoy(hyg_lqd_ratio).rename("hyg_lqd_ratio_yoy")

    vix_mom = pct_mom(vix_m).rename("vix_mom")
    vix_yoy = pct_yoy(vix_m).rename("vix_yoy")

    # Term structure
    vix_term = (vix3m_m - vix_m).rename("vix_term")

    # Signals (monthly analogs of your weekly rules)
    # - Trend: 10-month SMA on SPX close; ON if close >= SMA
    trend_sma_10m = spx_m.rolling(10, min_periods=10).mean().rename("trend_sma_10m")
    trend_on = (spx_m >= trend_sma_10m).astype(int).rename("trend_on")

    # - Credit: HYG/LQD ratio vs 10-month SMA; OFF if ratio < SMA
    credit_sma_10m = hyg_lqd_ratio.rolling(10, min_periods=10).mean().rename("credit_sma_10m")
    credit_off = (hyg_lqd_ratio < credit_sma_10m).astype(int).rename("credit_off")

    # - Vol: OFF if VIX > 25 or VIX > VIX3M
    vix_off = ((vix_m > 25.0) | (vix_m > vix3m_m)).astype(int).rename("vix_off")

    off_votes = (credit_off + vix_off).rename("off_votes")

    def classify(row):
        # Risk-On if trend_on == 1 and off_votes == 0
        # Risk-Off if off_votes >= 2
        # Mixed otherwise
        if row["trend_on"] == 1 and row["off_votes"] == 0:
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"

    # Assemble dataframe
    df = last_nonnull_join(
        [
            spx_m, sp500_mom, sp500_yoy, trend_sma_10m, trend_on,
            hyg_m, lqd_m, hyg_lqd_ratio, hyg_lqd_mom, hyg_lqd_yoy, credit_sma_10m, credit_off,
            vix_m, vix3m_m, vix_mom, vix_yoy, vix_term, vix_off,
            off_votes,
        ],
        how="outer",
    )

    df = df.sort_index()
    df.index.name = "date"

    # Classify regime
    df["regime"] = df[["trend_on", "off_votes"]].dropna().apply(classify, axis=1)

    # Clean up NaNs at the start (before we have enough history for SMAs)
    # Keep rows where at least the core levels exist so MoM/YoY can appear progressively.
    # When exporting, we’ll drop rows where everything is NaN in key columns.
    key_cols = ["sp500", "hyg_lqd_ratio", "vix", "vix3m", "trend_on", "credit_off", "vix_off", "off_votes", "regime"]
    df = df[df[key_cols].notna().any(axis=1)]

    # Ensure month-start index (already set) and save
    # CSV with ISO date string (YYYY-MM-01)
    out = df.reset_index()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    out = out[
        [
            "date",
            # SP500 + trend
            "sp500", "sp500_mom", "sp500_yoy", "trend_sma_10m", "trend_on",
            # Credit
            "hyg", "lqd", "hyg_lqd_ratio", "hyg_lqd_ratio_mom", "hyg_lqd_yoy", "credit_sma_10m", "credit_off",
            # Volatility
            "vix", "vix3m", "vix_mom", "vix_yoy", "vix_term", "vix_off",
            # Composite
            "off_votes", "regime",
        ]
    ]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(out):,} rows")

if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 200)
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)