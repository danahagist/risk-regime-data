# src/compute_risk_monthly.py
# Monthly risk pipeline (FRED-only):
# - Pulls SP500 (price level), HY OAS (bps), VIXCLS (level)
# - Converts daily -> monthly (month-end), then timestamps to month start (YYYY-MM-01)
# - Computes 10m SMA (SP500), 12m SMA (HY OAS), flags, MoM% and YoY%
# - Saves to data/risk_monthly_history.csv (appends + de-duplicates by date)

import os
import sys
import time
import math
import pandas as pd
from fredapi import Fred

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY is not set in the environment.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

OUT_PATH = "data/risk_monthly_history.csv"
START_DATE = "1990-01-01"

# FRED series
SPX_CODE = "SP500"             # S&P 500 index level (monthly from daily)
HY_OAS_CODE = "BAMLH0A0HYM2"   # ICE BofA US High Yield OAS (bps)
VIX_CODE = "VIXCLS"            # CBOE VIX Index (level)

# Parameters
TREND_SMA_MONTHS = 10
CREDIT_SMA_MONTHS = 12
CREDIT_OAS_BPS_THRESHOLD = 500.0
VIX_THRESHOLD = 25.0

# -------------------------------
# Utilities
# -------------------------------
def to_month_start_from_daily(s: pd.Series) -> pd.Series:
    """Daily -> month-end value, then stamp as month-start date."""
    if s.empty:
        return s
    s = s.sort_index()
    s = s.resample("M").last()  # take last obs each month
    s.index = s.index.to_period("M").to_timestamp("MS")  # set to 1st of month
    return s.dropna()

def to_mom_yoy_pct(s: pd.Series) -> (pd.Series, pd.Series):
    """Return MoM% and YoY% (percent changes) for a level series."""
    s = s.astype(float)
    mom = s.pct_change(1) * 100.0
    yoy = s.pct_change(12) * 100.0
    return mom, yoy

def safe_get_series(code: str, start: str) -> pd.Series:
    """Get series with simple retries; returns a Pandas Series indexed by DatetimeIndex."""
    last_err = None
    for i in range(3):
        try:
            s = fred.get_series(code, observation_start=start)
            s = pd.Series(s)
            s.index = pd.to_datetime(s.index)
            return s.dropna()
        except Exception as e:
            last_err = e
            wait = 2 * i
            print(f"[fredapi] attempt {i+1} failed for {code}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise last_err

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# -------------------------------
# Main
# -------------------------------
def main():
    # 1) Pull raw daily series
    spx_daily = safe_get_series(SPX_CODE, START_DATE)       # level
    hy_oas_daily = safe_get_series(HY_OAS_CODE, START_DATE) # bps
    vix_daily = safe_get_series(VIX_CODE, START_DATE)       # level

    # 2) Convert to monthly (month-end -> month-start)
    spx = to_month_start_from_daily(spx_daily).rename("sp500")
    hy_oas = to_month_start_from_daily(hy_oas_daily).rename("hy_oas")
    vix = to_month_start_from_daily(vix_daily).rename("vix")

    # 3) Build a common monthly index (inner join on months we have all three)
    df = pd.concat([spx, hy_oas, vix], axis=1).dropna().sort_index()

    # 4) Compute SMAs for signals (10m SPX, 12m HY OAS)
    df["sp500_sma_10m"] = df["sp500"].rolling(window=TREND_SMA_MONTHS, min_periods=TREND_SMA_MONTHS).mean()
    df["hy_oas_sma_12m"] = df["hy_oas"].rolling(window=CREDIT_SMA_MONTHS, min_periods=CREDIT_SMA_MONTHS).mean()

    # 5) Risk signal flags
    # Trend: ON if SPX >= 10m SMA (only when SMA available)
    df["trend_on"] = (df["sp500"] >= df["sp500_sma_10m"]).astype(int)

    # Credit: OFF if HY OAS > max(500 bps, 12m SMA)
    hy_barrier = pd.concat(
        [
            pd.Series(CREDIT_OAS_BPS_THRESHOLD, index=df.index),
            df["hy_oas_sma_12m"]
        ],
        axis=1
    ).max(axis=1)
    df["credit_off"] = (df["hy_oas"] > hy_barrier).astype(int)

    # Vol: OFF if VIX > 25
    df["vix_off"] = (df["vix"] > VIX_THRESHOLD).astype(int)

    # 6) Composite regime
    df["off_votes"] = df["credit_off"] + df["vix_off"]
    def _regime(row):
        if (row["trend_on"] == 1) and (row["off_votes"] == 0):
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"
    df["regime"] = df.apply(_regime, axis=1)

    # 7) MoM% and YoY% for each metric
    for col in ["sp500", "hy_oas", "vix"]:
        mom, yoy = to_mom_yoy_pct(df[col])
        df[f"{col}_mom"] = mom
        df[f"{col}_yoy"] = yoy

    # 8) Final formatting
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Columns order (clear and consistent)
    cols = [
        "date",
        "sp500", "sp500_sma_10m", "sp500_mom", "sp500_yoy", "trend_on",
        "hy_oas", "hy_oas_sma_12m", "hy_oas_mom", "hy_oas_yoy", "credit_off",
        "vix", "vix_mom", "vix_yoy", "vix_off",
        "off_votes", "regime"
    ]
    df = df[cols]

    # 9) Append + de-duplicate vs existing file (by date)
    ensure_dir(OUT_PATH)
    if os.path.exists(OUT_PATH):
        try:
            old = pd.read_csv(OUT_PATH, parse_dates=["date"])
            old["date"] = old["date"].dt.strftime("%Y-%m-%d")
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            combined.to_csv(OUT_PATH, index=False)
        except Exception as e:
            print(f"[WARN] Could not merge with existing {OUT_PATH}: {e}", file=sys.stderr)
            df.to_csv(OUT_PATH, index=False)
    else:
        df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH} with {len(df)} rows (latest date {df['date'].iloc[-1]})")

if __name__ == "__main__":
    main()