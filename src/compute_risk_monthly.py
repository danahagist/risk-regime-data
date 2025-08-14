# src/compute_risk_monthly.py
# Monthly risk pipeline (yfinance for market data):
# - Pulls SPY (proxy for S&P 500), HYG, LQD, ^VIX, ^VIX3M
# - Converts daily -> monthly (month-end), then timestamps to month start (YYYY-MM-01)
# - Computes 10m SMA (SPY), HYG/LQD ratio + 10m SMA, flags, MoM% and YoY%
# - Vol OFF if VIX>25 or VIX>VIX3M (if ^VIX3M available; otherwise only VIX>25)
# - Saves to data/risk_monthly_history.csv (appends + de-duplicates by date)

import os
import sys
import time
import math
import pandas as pd
import yfinance as yf

OUT_PATH = "data/risk_monthly_history.csv"
START_DATE = "1990-01-01"

# yfinance tickers
SPX_TICKER   = "SPY"      # proxy for S&P 500 trend
HYG_TICKER   = "HYG"
LQD_TICKER   = "LQD"
VIX_TICKER   = "^VIX"
VIX3M_TICKER = "^VIX3M"

# Parameters
TREND_SMA_MONTHS  = 10
CREDIT_SMA_MONTHS = 10
VIX_THRESHOLD     = 25.0

# -------------------------------
# Utilities
# -------------------------------
def to_month_start_from_daily(s: pd.Series) -> pd.Series:
    """Daily -> month-end value, then stamp as month-start date."""
    if s.empty:
        return s
    s = s.sort_index()
    # month-end; pandas warns 'M' is deprecated, use 'ME'
    s = s.resample("ME").last()
    # set to first of month
    s.index = s.index.to_period("M").to_timestamp("MS")
    return s.dropna()

def to_mom_yoy_pct(s: pd.Series) -> (pd.Series, pd.Series):
    """Return MoM% and YoY% (percent changes) for a level series."""
    s = s.astype(float)
    mom = s.pct_change(1) * 100.0
    yoy = s.pct_change(12) * 100.0
    return mom, yoy

def yf_series(ticker: str, start: str, max_retries: int = 4, pause: float = 0.0) -> pd.Series:
    """
    Download daily series from yfinance and return a single Series of prices.
    Prefers 'Adj Close' if available, else 'Close'.
    Retries with incremental backoff.
    """
    last_err = None
    for i in range(1, max_retries + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                progress=False,
                auto_adjust=False,
                threads=False,
                timeout=30
            )
            if df is None or df.empty:
                raise ValueError("Empty frame from Yahoo")

            col = None
            if "Adj Close" in df.columns:
                col = "Adj Close"
            elif "Close" in df.columns:
                col = "Close"
            else:
                raise ValueError("Neither 'Adj Close' nor 'Close' in dataframe")

            s = df[col].dropna()
            s.index = pd.to_datetime(s.index)
            s.name = ticker
            if s.empty:
                raise ValueError("Empty series after dropna")
            return s
        except Exception as e:
            last_err = e
            wait = 0 if i == 1 else [0, 2, 4, 6][i - 1]
            print(f"[yfinance] attempt {i} failed for {ticker}: {e} (sleep {wait}s)", file=sys.stderr)
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
    spy_daily   = yf_series(SPX_TICKER, START_DATE)   # proxy for S&P 500
    hyg_daily   = yf_series(HYG_TICKER, START_DATE)
    lqd_daily   = yf_series(LQD_TICKER, START_DATE)
    vix_daily   = yf_series(VIX_TICKER, START_DATE)

    # VIX3M optional
    try:
        vix3m_daily = yf_series(VIX3M_TICKER, START_DATE)
        have_vix3m = True
    except Exception as e:
        print(f"[WARN] Could not load {VIX3M_TICKER}: {e}. Falling back to VIX>25 rule only.", file=sys.stderr)
        vix3m_daily = pd.Series(dtype=float)
        have_vix3m = False

    # 2) Convert to monthly (month-end -> month-start)
    spx  = to_month_start_from_daily(spy_daily).rename("sp500")  # keep column name 'sp500'
    hyg  = to_month_start_from_daily(hyg_daily).rename("hyg")
    lqd  = to_month_start_from_daily(lqd_daily).rename("lqd")
    vix  = to_month_start_from_daily(vix_daily).rename("vix")
    if have_vix3m:
        vix3m = to_month_start_from_daily(vix3m_daily).rename("vix3m")
        df_vix = pd.concat([vix, vix3m], axis=1).dropna().sort_index()
    else:
        df_vix = vix.to_frame()

    # 3) Build a common monthly index for price/credit
    df_core = pd.concat([spx, hyg, lqd], axis=1).dropna().sort_index()

    # 4) Trend signal (10m SMA on SPY)
    df_core["sp500_sma_10m"] = df_core["sp500"].rolling(window=TREND_SMA_MONTHS, min_periods=TREND_SMA_MONTHS).mean()
    df_core["trend_on"] = (df_core["sp500"] >= df_core["sp500_sma_10m"]).astype(int)

    # 5) Credit ratio (HYG/LQD) and 10m SMA
    df_core["hyg_lqd_ratio"] = df_core["hyg"] / df_core["lqd"]
    df_core["hyg_lqd_ratio_sma_10m"] = df_core["hyg_lqd_ratio"].rolling(window=CREDIT_SMA_MONTHS, min_periods=CREDIT_SMA_MONTHS).mean()
    df_core["credit_off"] = (df_core["hyg_lqd_ratio"] < df_core["hyg_lqd_ratio_sma_10m"]).astype(int)

    # 6) Vol OFF: VIX>25 OR (if available) VIX>VIX3M
    if have_vix3m:
        df_core = df_core.join(df_vix, how="inner")
        df_core["vix_off"] = ((df_core["vix"] > VIX_THRESHOLD) | (df_core["vix"] > df_core["vix3m"])).astype(int)
    else:
        df_core = df_core.join(df_vix, how="inner")
        df_core["vix_off"] = (df_core["vix"] > VIX_THRESHOLD).astype(int)

    # 7) Composite regime
    df_core["off_votes"] = df_core["credit_off"] + df_core["vix_off"]

    def _regime(row):
        if (row["trend_on"] == 1) and (row["off_votes"] == 0):
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"

    df_core["regime"] = df_core.apply(_regime, axis=1)

    # 8) MoM% and YoY% for metrics we display
    for col in ["sp500", "hyg_lqd_ratio", "vix"]:
        mom, yoy = to_mom_yoy_pct(df_core[col])
        df_core[f"{col}_mom"] = mom
        df_core[f"{col}_yoy"] = yoy

    # 9) Final formatting
    df = df_core.reset_index().rename(columns={"index": "date"})
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Column order (kept consistent with earlier file; sp500 is SPY proxy)
    cols = [
        "date",
        "sp500", "sp500_sma_10m", "sp500_mom", "sp500_yoy", "trend_on",
        "hyg", "lqd", "hyg_lqd_ratio", "hyg_lqd_ratio_sma_10m", "hyg_lqd_ratio_mom", "hyg_lqd_ratio_yoy", "credit_off",
        "vix",
    ]
    if have_vix3m:
        cols += ["vix3m"]
    cols += ["vix_mom", "vix_yoy", "vix_off", "off_votes", "regime"]

    df = df[cols]

    # 10) Append + de-duplicate vs existing file (by date)
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