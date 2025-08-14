# src/compute_risk_monthly.py
# Risk signals via Yahoo Finance (HYG/LQD, ^GSPC, ^VIX, ^VIX3M)
# - Pull daily data with yfinance
# - Aggregate to monthly (month-end), then set date to month-start (YYYY-MM-01)
# - Compute: 10m SMA for ^GSPC trend, 12m SMA for HYG/LQD ratio (credit),
#            VIX term-structure OFF if VIX > 25 or VIX > VIX3M
# - Also compute MoM% and YoY% for sp500, hyg_lqd_ratio, and vix
# - Append + de-duplicate by date to data/risk_monthly_history.csv

import os
import sys
import time
import math
import pandas as pd
import yfinance as yf

OUT_PATH = "data/risk_monthly_history.csv"
START_DATE = "1990-01-01"

# Yahoo tickers
SPX_TICKER   = "^GSPC"   # S&P 500 index
HYG_TICKER   = "HYG"     # iShares iBoxx $ High Yield Corp Bond ETF
LQD_TICKER   = "LQD"     # iShares iBoxx $ Investment Grade Corp Bond ETF
VIX_TICKER   = "^VIX"    # VIX
VIX3M_TICKER = "^VIX3M"  # 3M VIX

# Parameters
TREND_SMA_MONTHS   = 10
CREDIT_SMA_MONTHS  = 12
CREDIT_OAS_FLOOR   = None   # not used for ETF ratio (kept for parity)
VIX_THRESHOLD      = 25.0

# -------------------------------
# Helpers
# -------------------------------

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _best_close(df: pd.DataFrame) -> pd.Series:
    # Prefer Adjusted Close if present; fallback to Close
    if "Adj Close" in df.columns:
        return df["Adj Close"]
    if "Close" in df.columns:
        return df["Close"]
    # Handle multi-index columns from multi-ticker download
    if ("Adj Close" in df.columns.get_level_values(0)):
        return df["Adj Close"].iloc[:, 0]
    if ("Close" in df.columns.get_level_values(0)):
        return df["Close"].iloc[:, 0]
    raise ValueError("No Close/Adj Close found")

def yf_series(ticker: str, start: str) -> pd.Series:
    """Download a single ticker with retries; return daily close series."""
    last_err = None
    for i in range(4):
        try:
            df = yf.download(ticker, start=start, progress=False, auto_adjust=False, threads=False)
            if df is None or df.empty:
                raise ValueError("Empty frame from Yahoo")
            s = _best_close(df).copy()
            s.index = pd.to_datetime(s.index)
            s.name = ticker
            return s.dropna()
        except Exception as e:
            last_err = e
            wait = 2 * i
            print(f"[yfinance] attempt {i+1} failed for {ticker}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise last_err

def to_month_start_from_daily(s: pd.Series) -> pd.Series:
    """Daily -> month-end value, then stamp as month-start date."""
    if s.empty:
        return s
    s = s.sort_index()
    s = s.resample("ME").last()  # month-end (use 'ME' to avoid deprecation)
    s.index = s.index.to_period("M").to_timestamp("MS")  # first of month
    return s.dropna()

def mom_yoy_pct(s: pd.Series):
    s = s.astype(float)
    return s.pct_change(1) * 100.0, s.pct_change(12) * 100.0

# -------------------------------
# Main
# -------------------------------

def main():
    # 1) Pull daily from Yahoo
    spx_daily   = yf_series(SPX_TICKER, START_DATE)     # index level
    hyg_daily   = yf_series(HYG_TICKER, START_DATE)     # ETF price
    lqd_daily   = yf_series(LQD_TICKER, START_DATE)     # ETF price
    vix_daily   = yf_series(VIX_TICKER, START_DATE)     # index level
    vix3m_daily = yf_series(VIX3M_TICKER, START_DATE)   # index level

    # 2) Daily -> monthly (month-end), then set index to month-start
    spx   = to_month_start_from_daily(spx_daily).rename("sp500")
    hyg   = to_month_start_from_daily(hyg_daily).rename("hyg")
    lqd   = to_month_start_from_daily(lqd_daily).rename("lqd")
    vix   = to_month_start_from_daily(vix_daily).rename("vix")
    vix3m = to_month_start_from_daily(vix3m_daily).rename("vix3m")

    # 3) Build combined monthly df (inner join on months with all series)
    df = pd.concat([spx, hyg, lqd, vix, vix3m], axis=1).dropna().sort_index()

    # 4) Credit metric: HYG/LQD ratio
    df["hyg_lqd_ratio"] = df["hyg"] / df["lqd"]

    # 5) SMAs
    df["sp500_sma_10m"]     = df["sp500"].rolling(TREND_SMA_MONTHS, min_periods=TREND_SMA_MONTHS).mean()
    df["hyg_lqd_sma_12m"]   = df["hyg_lqd_ratio"].rolling(CREDIT_SMA_MONTHS, min_periods=CREDIT_SMA_MONTHS).mean()

    # 6) Risk signal flags
    # Trend: ON if SPX >= 10m SMA (where SMA exists)
    df["trend_on"] = (df["sp500"] >= df["sp500_sma_10m"]).astype(int)

    # Credit: OFF if HYG/LQD ratio < its 12m SMA (risk appetite deteriorating)
    df["credit_off"] = (df["hyg_lqd_ratio"] < df["hyg_lqd_sma_12m"]).astype(int)

    # Vol: OFF if VIX > 25 or VIX > VIX3M
    df["vix_off"] = ((df["vix"] > VIX_THRESHOLD) | (df["vix"] > df["vix3m"])).astype(int)

    # 7) Composite regime
    df["off_votes"] = df["credit_off"] + df["vix_off"]
    def _regime(row):
        if (row["trend_on"] == 1) and (row["off_votes"] == 0):
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"
    df["regime"] = df.apply(_regime, axis=1)

    # 8) MoM% and YoY% for displayed metrics
    for col in ["sp500", "hyg_lqd_ratio", "vix"]:
        mom, yoy = mom_yoy_pct(df[col])
        df[f"{col}_mom"] = mom
        df[f"{col}_yoy"] = yoy

    # 9) Final formatting
    df_out = df.reset_index().rename(columns={"index": "date"})
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")

    cols = [
        "date",
        "sp500", "sp500_sma_10m", "sp500_mom", "sp500_yoy", "trend_on",
        "hyg_lqd_ratio", "hyg_lqd_sma_12m", "hyg_lqd_ratio_mom", "hyg_lqd_ratio_yoy", "credit_off",
        "vix", "vix_mom", "vix_yoy", "vix3m", "vix_off",
        "off_votes", "regime"
    ]
    df_out = df_out[cols]

    # 10) Append + de-duplicate
    ensure_dir(OUT_PATH)
    if os.path.exists(OUT_PATH):
        try:
            old = pd.read_csv(OUT_PATH, parse_dates=["date"])
            old["date"] = old["date"].dt.strftime("%Y-%m-%d")
            combined = pd.concat([old, df_out], ignore_index=True)
            combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            combined.to_csv(OUT_PATH, index=False)
        except Exception as e:
            print(f"[WARN] Could not merge with existing {OUT_PATH}: {e}", file=sys.stderr)
            df_out.to_csv(OUT_PATH, index=False)
    else:
        df_out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH} with {len(df_out)} rows (latest {df_out['date'].iloc[-1]})")

if __name__ == "__main__":
    main()