# src/compute_macro.py
import os, sys, time, json, math
from datetime import datetime
import pandas as pd
from fredapi import Fred
import requests

START = "1970-01-01"
OUTFILE = "data/macro_weekly_history.csv"

API_KEY = os.getenv("FRED_API_KEY", "").strip()
if not API_KEY:
    print("ERROR: FRED_API_KEY is not set. Add it as a GitHub Secret and pass it in the workflow env.")
    sys.exit(1)

fred = Fred(api_key=API_KEY)

# ---- Indicator catalog ----
# code: {name, transform}
# transform: one of {None, "yoy", "mom", "diff"} applied BEFORE weekly alignment
INDICATORS = {
    # Your 3 regime drivers
    "NAPM":        {"name": "ISM_Manufacturing_PMI", "transform": None},          # level (>50 expansion)
    "USSLIND":     {"name": "LEI_Smoothed_Diffs",    "transform": None},          # already a change index
    "T10Y3M":      {"name": "YieldCurve_10Y_minus_3M","transform": None},         # spread (bps)

    # Core “market health” dashboard examples
    "PAYEMS":      {"name": "Nonfarm_Payrolls",      "transform": "diff"},        # monthly change
    "UNRATE":      {"name": "Unemployment_Rate",     "transform": None},          # level
    "CPIAUCSL":    {"name": "CPI_All_Items_YoY",     "transform": "yoy"},         # YoY %
    "PCEPILFE":    {"name": "Core_PCE_YoY",          "transform": "yoy"},         # YoY %
    "INDPRO":      {"name": "Industrial_Production_YoY","transform": "yoy"},      # YoY %
    "PERMIT":      {"name": "Housing_Permits_MoM",   "transform": "mom"},         # MoM %
    "RSAFS":       {"name": "Retail_Sales_Advance_MoM","transform": "mom"},       # MoM %
}

def get_series(code: str, start: str = START, retries: int = 3, backoff: float = 2.0) -> pd.Series:
    last_err = None
    for i in range(retries):
        try:
            s = fred.get_series(code, observation_start=start)
            s = pd.Series(s).dropna()
            s.index = pd.to_datetime(s.index)
            s.name = code
            print(f"[fredapi] {code}: {len(s)} rows")
            return s
        except (requests.exceptions.RequestException, Exception) as e:
            last_err = e
            wait = backoff * (i + 1)
            print(f"[fredapi] attempt {i+1} failed for {code}: {e} (sleep {wait}s)")
            time.sleep(wait)
    raise last_err

def apply_transform(s: pd.Series, transform: str | None) -> pd.Series:
    if transform is None:
        return s
    if transform == "yoy":
        # YoY percent change (assumes monthly/native cadence)
        return s.pct_change(12) * 100.0
    if transform == "mom":
        return s.pct_change(1) * 100.0
    if transform == "diff":
        return s.diff(1)
    raise ValueError(f"Unknown transform: {transform}")

def align_weekly_last(s: pd.Series) -> pd.Series:
    """
    Align to weekly Fridays using last available observation then forward-fill.
    Limit ffill to 8 weeks to avoid carrying stale values indefinitely.
    """
    # Resample to weekly Friday with last known point as of that week
    # We first asfreq to daily via bfill/ffill to avoid gaps, then take Friday
    s_weekly = s.resample("W-FRI").last()
    # If a month has not reported yet, forward-fill for a short window
    s_weekly = s_weekly.ffill(limit=8)
    return s_weekly

def main():
    os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)

    weekly_cols = {}
    for code, meta in INDICATORS.items():
        raw = get_series(code, START)
        transformed = apply_transform(raw, meta["transform"]).dropna()
        weekly = align_weekly_last(transformed).rename(meta["name"])
        weekly_cols[meta["name"]] = weekly

    # Combine all weekly series
    df = pd.concat(weekly_cols.values(), axis=1).sort_index()
    # Keep only rows where we have at least one value; optional threshold:
    df = df.dropna(how="all")

    # Write (create or overwrite) weekly history
    # If you'd prefer append semantics, switch to the append/merge pattern.
    df_out = df.copy()
    df_out.index.name = "date"
    df_out.reset_index(inplace=True)
    df_out["date"] = df_out["date"].dt.date.astype(str)

    df_out.to_csv(OUTFILE, index=False)
    print(f"Wrote {OUTFILE} with {len(df_out)} weekly rows and {len(df_out.columns)-1} indicators.")

if __name__ == "__main__":
    main()