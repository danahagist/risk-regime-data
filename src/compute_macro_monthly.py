# src/compute_macro_monthly.py
"""
Build a monthly macro dataset from FRED (no yfinance).
- Daily series -> monthly median, then MoM/YoY
- Monthly series -> aligned to month start, MoM/YoY
- Quarterly series -> QoQ/YoY at quarterly, then forward-filled to months
- Dates truncated to month-start (YYYY-MM-01)
- Core-3 signals: INDPRO YoY > 0, USSLIND YoY > 0, Yield Curve (10y-3m) > 0 bps
Outputs: data/macro_monthly_history.csv
"""

import os
import sys
from datetime import datetime
import pandas as pd
from fredapi import Fred

# ----------------------------
# Config
# ----------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY") or os.getenv("FRED_API_TOKEN") or os.getenv("FRED_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set in environment.", file=sys.stderr)
    sys.exit(1)

START = "1980-01-01"  # change if you want earlier history
OUTPUT_PATH = "data/macro_monthly_history.csv"

# FRED series codes
CODES_DAILY = {
    "DGS10": "dgs10",                # 10Y Treasury (percent)
    "DGS3MO": "dgs3mo",              # 3M Treasury (percent)
    "BAMLH0A0HYM2": "hy_oas",        # HY OAS (percent points)
    "SP500": "sp500",                # S&P 500 index level
}

CODES_MONTHLY = {
    "INDPRO": "indpro",              # Industrial Production index
    "USSLIND": "lei",                # Leading Index
    "UNRATE": "unrate",              # Unemployment rate (%)
    "CPIAUCSL": "cpi",               # CPI (index, SA)
    "UMCSENT": "umcsent",            # Consumer Sentiment
    "HOUST": "houst",                # Housing Starts (annualized units)
}

CODES_QUARTERLY = {
    "A191RL1Q225SBEA": "gdp_saar",   # Real GDP, QoQ SAAR (%)
}

# ----------------------------
# Helpers
# ----------------------------
def fetch_fred_series(fred: Fred, code: str, start: str) -> pd.Series:
    """Fetch a FRED series as a pandas Series with DatetimeIndex."""
    s = fred.get_series(code, observation_start=start)
    s = pd.Series(s)
    s.index = pd.to_datetime(s.index)
    s.name = code
    return s.sort_index()

def to_month_start(s: pd.Series) -> pd.Series:
    """Map index to month-start (YYYY-MM-01), collapse duplicates by last observation."""
    s = s.dropna()
    s.index = s.index.to_period("M").to_timestamp("MS")
    s = s.groupby(level=0).last().sort_index()
    return s

def daily_to_monthly_median(s: pd.Series) -> pd.Series:
    """Daily series → monthly median at month start."""
    s = s.dropna().sort_index()
    m = s.resample("MS").median()
    return m

def compute_mom_yoy_monthly(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """MoM and YoY percent change on a monthly series."""
    s = s.sort_index()
    mom = s.pct_change(1) * 100.0
    yoy = s.pct_change(12) * 100.0
    return mom, yoy

def quarterly_to_months_with_changes(s: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Quarterly series → keep quarterly values, compute QoQ and YoY at quarterly,
    then forward-fill each within the quarter to monthly month-start index.
    Returns (monthly_value, monthly_qoq, monthly_yoy)
    """
    s = s.dropna().sort_index()
    # normalize index to quarter start (Q-S), then compute changes at quarterly freq
    q = s.copy()
    q.index = q.index.to_period("Q").to_timestamp("QS")
    q = q.groupby(level=0).last().sort_index()

    qoq = q.pct_change(1) * 100.0
    yoy = q.pct_change(4) * 100.0

    # Expand to monthly month-start with forward-fill
    monthly_index = pd.date_range(q.index.min(), datetime.today(), freq="MS")
    q_m = q.reindex(monthly_index, method="ffill")
    qoq_m = qoq.reindex(monthly_index, method="ffill")
    yoy_m = yoy.reindex(monthly_index, method="ffill")
    return q_m, qoq_m, yoy_m

def safe_join(df: pd.DataFrame, s: pd.Series, colname: str) -> pd.DataFrame:
    if s is None or s.empty:
        df[colname] = pd.NA
        return df
    s = s.rename(colname)
    return df.join(s, how="outer")

# ----------------------------
# Main build
# ----------------------------
def main():
    fred = Fred(api_key=FRED_API_KEY)

    # Container for monthly-aligned outputs
    monthly = pd.DataFrame()

    # ---------- Daily series → monthly median ----------
    daily_series = {}
    for code, col in CODES_DAILY.items():
        s = fetch_fred_series(fred, code, START)
        m = daily_to_monthly_median(s)
        daily_series[col] = m

    # join daily (monthly-aggregated) into frame
    for col, s in daily_series.items():
        monthly = safe_join(monthly, s, col)

    # ---------- Monthly series (align to month start) ----------
    monthly_series = {}
    for code, col in CODES_MONTHLY.items():
        s = fetch_fred_series(fred, code, START)
        m = to_month_start(s)
        monthly_series[col] = m

    for col, s in monthly_series.items():
        monthly = safe_join(monthly, s, col)

    # ---------- Quarterly series (expand to months with QoQ/YoY) ----------
    # We’ll compute QoQ/YoY for quarterly series at quarterly frequency
    # then forward-fill within quarter to months.
    q_map = {}
    for code, col in CODES_QUARTERLY.items():
        s = fetch_fred_series(fred, code, START)
        val_m, qoq_m, yoy_m = quarterly_to_months_with_changes(s)
        q_map[col] = (val_m, qoq_m, yoy_m)

    # core index & sort
    monthly.index.name = "date"
    monthly = monthly.sort_index()

    # Join quarterly
    for col, (val_m, qoq_m, yoy_m) in q_map.items():
        monthly = safe_join(monthly, val_m, col)          # e.g., gdp_saar
        monthly = safe_join(monthly, qoq_m, f"{col}_qoq") # e.g., gdp_saar_qoq
        monthly = safe_join(monthly, yoy_m, f"{col}_yoy") # e.g., gdp_saar_yoy

    # Keep only month-start index
    monthly = monthly[~monthly.index.duplicated(keep="last")]
    monthly = monthly.asfreq("MS")

    # ----------------------------
    # Changes (MoM/YoY) for daily→monthly & monthly series
    # ----------------------------
    # INDPRO + signals
    if "indpro" in monthly:
        monthly["indpro_mom"], monthly["indpro_yoy"] = compute_mom_yoy_monthly(monthly["indpro"])
        monthly["indpro_signal"] = (monthly["indpro_yoy"] > 0).astype("Int64")

    # LEI + signals
    if "lei" in monthly:
        monthly["lei_mom"], monthly["lei_yoy"] = compute_mom_yoy_monthly(monthly["lei"])
        monthly["lei_signal"] = (monthly["lei_yoy"] > 0).astype("Int64")

    # Yield curve (bps) + signals
    if {"dgs10", "dgs3mo"}.issubset(monthly.columns):
        # Both are in percent; convert spread to basis points
        monthly["yield_curve_spread"] = (monthly["dgs10"] - monthly["dgs3mo"]) * 100.0
        # MoM/YoY on the spread itself
        monthly["yield_curve_mom"], monthly["yield_curve_yoy"] = compute_mom_yoy_monthly(monthly["yield_curve_spread"])
        monthly["yield_curve_signal"] = (monthly["yield_curve_spread"] > 0).astype("Int64")

    # HY OAS (percent points)
    if "hy_oas" in monthly:
        monthly["hy_oas_mom"], monthly["hy_oas_yoy"] = compute_mom_yoy_monthly(monthly["hy_oas"])

    # Unemployment (%)
    if "unrate" in monthly:
        monthly["unrate_mom"], monthly["unrate_yoy"] = compute_mom_yoy_monthly(monthly["unrate"])

    # CPI (index) → changes in %
    if "cpi" in monthly:
        monthly["cpi_mom"], monthly["cpi_yoy"] = compute_mom_yoy_monthly(monthly["cpi"])

    # UMCSENT
    if "umcsent" in monthly:
        monthly["umcsent_mom"], monthly["umcsent_yoy"] = compute_mom_yoy_monthly(monthly["umcsent"])

    # HOUST
    if "houst" in monthly:
        monthly["houst_mom"], monthly["houst_yoy"] = compute_mom_yoy_monthly(monthly["houst"])

    # SP500
    if "sp500" in monthly:
        monthly["sp500_mom"], monthly["sp500_yoy"] = compute_mom_yoy_monthly(monthly["sp500"])

    # Placeholder for earnings yield (we do not have earnings in FRED here)
    monthly["sp500_earnings_yield"] = pd.NA

    # ----------------------------
    # Core-3 Macro Score / Regime
    # ----------------------------
    sig_cols = []
    for c in ["indpro_signal", "lei_signal", "yield_curve_signal"]:
        if c in monthly:
            sig_cols.append(c)

    if sig_cols:
        monthly["macro_score"] = monthly[sig_cols].sum(axis=1).astype("Int64")
        monthly["macro_regime"] = monthly["macro_score"].apply(lambda x: "expansion" if pd.notna(x) and x >= 2 else ("contraction" if pd.notna(x) else pd.NA))

    # ----------------------------
    # Final tidy & write
    # ----------------------------
    monthly = monthly.reset_index()
    monthly["date"] = monthly["date"].dt.strftime("%Y-%m-%d")  # ensure month-start string

    # Recommended column order (adjust as you like)
    col_order = [
        "date",
        # Core 3 + signals
        "indpro", "indpro_mom", "indpro_yoy", "indpro_signal",
        "lei", "lei_mom", "lei_yoy", "lei_signal",
        "dgs10", "dgs3mo",
        "yield_curve_spread", "yield_curve_mom", "yield_curve_yoy", "yield_curve_signal",
        "macro_score", "macro_regime",
        # Additional indicators
        "hy_oas", "hy_oas_mom", "hy_oas_yoy",
        "unrate", "unrate_mom", "unrate_yoy",
        "cpi", "cpi_mom", "cpi_yoy",
        "gdp_saar", "gdp_saar_qoq", "gdp_saar_yoy",
        "umcsent", "umcsent_mom", "umcsent_yoy",
        "houst", "houst_mom", "houst_yoy",
        "sp500", "sp500_mom", "sp500_yoy", "sp500_earnings_yield",
    ]
    # keep only columns that exist
    col_order = [c for c in col_order if c in monthly.columns]
    monthly = monthly[col_order]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} with {len(monthly)} rows and {len(monthly.columns)} columns.")

if __name__ == "__main__":
    main()