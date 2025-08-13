# src/compute_macro.py
# FRED-only pipeline: Macro Regime Core 3 + dashboard metrics (weekly output)

import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from fredapi import Fred

# -----------------------------
# Config
# -----------------------------

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY environment variable is not set.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

START = "1990-01-01"
OUT_PATH = "data/macro_weekly_history.csv"
WEEKLY_FREQ = "W-FRI"

# -----------------------------
# Series (exact IDs as requested)
# -----------------------------

# Core 3 (PMI replaced by INDPRO, per decision)
INDPRO = "INDPRO"          # Industrial Production Index (monthly)
USSLIND = "USSLIND"        # Leading Index for the U.S. (monthly)
DGS10 = "DGS10"            # 10Y Treasury Constant Maturity Rate (daily, %)
DGS3MO = "DGS3MO"          # 3M Treasury Constant Maturity Rate (daily, %)

# Dashboard
BAML_HY_OAS = "BAMLH0A0HYM2"  # High Yield OAS (daily, %)
UNRATE = "UNRATE"              # Unemployment Rate (monthly, %)
CPI = "CPIAUCSL"               # CPI (monthly, SA index) -> YoY
GDP_GROWTH = "A191RL1Q225SBEA" # Real GDP, % change from preceding period, SAAR (quarterly, %)
UMCSENT = "UMCSENT"            # Consumer Sentiment (monthly, index)
HOUST = "HOUST"                # Housing Starts (monthly, SAAR, thousands)
SP500 = "SP500"                # S&P 500 index level (daily)

# Smoothing choices
YIELD_SPREAD_SMOOTH_DAYS = 5   # small smoothing on daily series before weekly mapping
INDPRO_YOY_SMOOTH_M = 3        # months
LEI_YOY_SMOOTH_M = 3           # months

# -----------------------------
# Helpers
# -----------------------------

def retry_call(fn, retries=3, sleep_secs=2):
    last_err: Optional[Exception] = None
    for i in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if i == retries:
                raise
            time.sleep(sleep_secs)
    if last_err:
        raise last_err

def get_series(code: str, start: str) -> pd.Series:
    """Fetch a FRED series as a pandas Series with DatetimeIndex."""
    def _fetch():
        s = fred.get_series(code, observation_start=start)
        if not isinstance(s.index, pd.DatetimeIndex):
            s.index = pd.to_datetime(s.index)
        try:
            s.index = s.index.tz_localize(None)
        except Exception:
            pass
        s.name = code
        return s
    return retry_call(_fetch)

def to_weekly_ffill(s: pd.Series, weekly_index: pd.DatetimeIndex) -> pd.Series:
    """Map any frequency to Friday-weekly via daily asfreq and forward-fill."""
    s_daily = s.asfreq("D")
    out = s_daily.reindex(weekly_index).ffill()
    out.name = s.name
    return out

def weekly_index(start: str, end: Optional[pd.Timestamp] = None) -> pd.DatetimeIndex:
    start_dt = pd.to_datetime(start)
    end_dt = (pd.Timestamp.today().normalize() if end is None else pd.to_datetime(end))
    return pd.date_range(start_dt, end_dt, freq=WEEKLY_FREQ)

# -----------------------------
# Core 3 signals
# -----------------------------

def signal_indpro(weekly_ix: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """INDPRO YoY (smoothed); signal=1 if YoY>0."""
    s = get_series(INDPRO, START).dropna()
    yoy = s.pct_change(12)
    yoy_sm = yoy.rolling(INDPRO_YOY_SMOOTH_M, min_periods=1).mean()
    sig_m = (yoy_sm > 0).astype(int)
    sig_w = to_weekly_ffill(sig_m, weekly_ix).rename("indpro_signal")
    yoy_w = to_weekly_ffill(yoy_sm, weekly_ix).rename("indpro_yoy_sm")
    level_w = to_weekly_ffill(s, weekly_ix).rename("indpro")
    return sig_w, yoy_w, level_w

def signal_lei(weekly_ix: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series]:
    """USSLIND YoY (smoothed); signal=1 if YoY>0."""
    s = get_series(USSLIND, START).dropna()
    yoy = s.pct_change(12)
    yoy_sm = yoy.rolling(LEI_YOY_SMOOTH_M, min_periods=1).mean()
    sig_m = (yoy_sm > 0).astype(int)
    sig_w = to_weekly_ffill(sig_m, weekly_ix).rename("lei_signal")
    yoy_w = to_weekly_ffill(yoy_sm, weekly_ix).rename("lei_yoy_sm")
    level_w = to_weekly_ffill(s, weekly_ix).rename("lei")
    return sig_w, yoy_w, level_w

def signal_yield_curve(weekly_ix: pd.DatetimeIndex) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Yield curve = DGS10 - DGS3MO; signal=1 if spread>0."""
    t10 = get_series(DGS10, START).dropna()      # %
    t3m = get_series(DGS3MO, START).dropna()     # %
    # Smooth slightly at daily granularity
    t10_sm = t10.rolling(YIELD_SPREAD_SMOOTH_DAYS, min_periods=1).mean()
    t3m_sm = t3m.rolling(YIELD_SPREAD_SMOOTH_DAYS, min_periods=1).mean()
    spread = (t10_sm - t3m_sm)
    sig_d = (spread > 0).astype(int)

    # Weekly mapping
    spread_w = to_weekly_ffill(spread.rename("yield_curve_spread"), weekly_ix)
    sig_w = to_weekly_ffill(sig_d.rename("yield_curve_signal"), weekly_ix).astype(int)
    t10_w = to_weekly_ffill(t10_sm.rename("dgs10"), weekly_ix)
    t3m_w = to_weekly_ffill(t3m_sm.rename("dgs3mo"), weekly_ix)
    return sig_w, spread_w, t10_w, t3m_w

# -----------------------------
# Dashboard metrics
# -----------------------------

def dashboard_metrics(weekly_ix: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=weekly_ix)

    # 4) High Yield OAS (daily, %) -> weekly ffill
    hy = get_series(BAML_HY_OAS, START).dropna()
    out["hy_oas"] = to_weekly_ffill(hy, weekly_ix)

    # 5) Unemployment rate (monthly, %) -> weekly ffill
    un = get_series(UNRATE, START).dropna()
    out["unrate"] = to_weekly_ffill(un, weekly_ix)

    # 6) CPI YoY (from CPIAUCSL, monthly index) -> pct_change(12) * 100
    cpi = get_series(CPI, START).dropna()
    cpi_yoy = cpi.pct_change(12) * 100.0
    out["cpi_yoy"] = to_weekly_ffill(cpi_yoy.rename("cpi_yoy"), weekly_ix)

    # 7) GDP Growth (already % SAAR, quarterly) -> weekly ffill
    gdp = get_series(GDP_GROWTH, START).dropna()
    out["gdp_saar"] = to_weekly_ffill(gdp.rename("gdp_saar"), weekly_ix)

    # 8) Consumer Confidence (monthly index) -> weekly ffill
    um = get_series(UMCSENT, START).dropna()
    out["umcsent"] = to_weekly_ffill(um, weekly_ix)

    # 9) Housing Starts (monthly SAAR, thousands) -> weekly ffill
    hs = get_series(HOUST, START).dropna()
    out["houst"] = to_weekly_ffill(hs, weekly_ix)

    # 10) SP500 (daily) + placeholder Earnings Yield
    spx = get_series(SP500, START).dropna()
    out["sp500"] = to_weekly_ffill(spx, weekly_ix)

    # Earnings yield requires earnings; we only have SP500 per spec -> leave NaN
    out["sp500_earnings_yield"] = np.nan
    print("NOTE: 'sp500_earnings_yield' left NaN (needs earnings).", file=sys.stderr)

    return out

# -----------------------------
# Main
# -----------------------------

def main():
    w_ix = weekly_index(START)

    # Core 3
    indpro_sig, indpro_yoy_w, indpro_level_w = signal_indpro(w_ix)
    lei_sig, lei_yoy_w, lei_level_w = signal_lei(w_ix)
    curve_sig, curve_spread_w, dgs10_w, dgs3m_w = signal_yield_curve(w_ix)

    signals = pd.concat(
        [indpro_sig.rename("indpro"), lei_sig.rename("lei"), curve_sig.rename("curve")],
        axis=1
    ).dropna(how="any")
    
    # Sum across columns to get a 0–3 macro score (ints)
    macro_score = signals.sum(axis=1).astype("int64").rename("macro_score")

    # Assemble core frame
    core = pd.DataFrame(index=w_ix)
    core["indpro"] = indpro_level_w
    core["indpro_yoy_sm"] = indpro_yoy_w
    core["indpro_signal"] = indpro_sig

    core["lei"] = lei_level_w
    core["lei_yoy_sm"] = lei_yoy_w
    core["lei_signal"] = lei_sig

    core["dgs10"] = dgs10_w
    core["dgs3mo"] = dgs3m_w
    core["yield_curve_spread"] = curve_spread_w
    core["yield_curve_signal"] = curve_sig

    core["macro_score"] = macro_score

    # Dashboard
    dash = dashboard_metrics(w_ix)

    out = core.join(dash, how="outer").dropna(how="all")
    out = out.sort_index()

    # Ensure output dir
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    # Merge with prior if exists
    if os.path.exists(OUT_PATH):
        prev = pd.read_csv(OUT_PATH, parse_dates=["date"]).set_index("date")
        merged = prev.combine_first(out)
        merged.update(out)
        merged = merged.sort_index()
        merged.to_csv(OUT_PATH, index_label="date")
    else:
        out.to_csv(OUT_PATH, index_label="date")

    # Log the latest snapshot
    last = out.dropna().tail(1)
    if not last.empty:
        dt = last.index[-1].date().isoformat()
        print(
            f"wrote {OUT_PATH} through {dt} | "
            f"Core signals (indpro/lei/curve)="
            f"{int(last['indpro_signal'].iloc[0])}/"
            f"{int(last['lei_signal'].iloc[0])}/"
            f"{int(last['yield_curve_signal'].iloc[0])} | "
            f"score={int(last['macro_score'].iloc[0])}"
        )

if __name__ == "__main__":
    main()
