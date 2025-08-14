# src/compute_macro_monthly.py
# Combined Macro + Risk (FRED-only), monthly at month-start dates.
# - Macro Core 3: INDPRO YoY>0, USSLIND YoY>0, yield curve (DGS10-DGS3MO)>0
# - Additional macro: HY OAS, UNRATE, CPI YoY, GDP SAAR (quarterly->monthly ffill), UMCSENT, HOUST, SP500 level
# - Risk: SP500 10m trend, HY OAS vs max(500bps, 12m SMA), VIX>25; off_votes, risk_regime
# - Also include MoM/YoY for SP500, HY OAS, VIX (convenient for UI)
# - Output: data/macro_monthly_history.csv (append + dedupe by date)

import os
import sys
import time
import numpy as np
import pandas as pd
from fredapi import Fred

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

OUT_PATH = "data/macro_monthly_history.csv"
START = "1990-01-01"

# ----- FRED series codes -----
# Macro core
INDPRO = "INDPRO"        # Industrial Production: Total Index (monthly)
USSLIND = "USSLIND"      # Leading Index for the United States (monthly)
DGS10 = "DGS10"          # 10-Year Treasury Constant Maturity Rate (daily)
DGS3MO = "DGS3MO"        # 3-Month Treasury Constant Maturity Rate (daily)

# Additional macro
HY_OAS = "BAMLH0A0HYM2"  # ICE BofA US High Yield OAS (daily, bps)
UNRATE = "UNRATE"        # Civilian Unemployment Rate (monthly, %)
CPI = "CPIAUCSL"         # CPI (monthly, index)
GDP_SAAR = "A191RL1Q225SBEA"  # Real GDP QoQ SAAR (%) (quarterly)
UMCSENT = "UMCSENT"      # U. Michigan Consumer Sentiment (monthly)
HOUST = "HOUST"          # Housing Starts (monthly)
SP500 = "SP500"          # S&P 500 index level (daily)
VIXCLS = "VIXCLS"        # CBOE VIX index (daily)

# ----- Risk params -----
TREND_SMA_MONTHS = 10
CREDIT_SMA_MONTHS = 12
CREDIT_OAS_BPS_THRESHOLD = 500.0
VIX_THRESHOLD = 25.0

# ===================== Utilities =====================

def safe_get_series(code: str, start: str) -> pd.Series:
    """Get FRED series with retries, returning a float series with DatetimeIndex."""
    last_err = None
    for i in range(3):
        try:
            s = fred.get_series(code, observation_start=start)
            s = pd.Series(s).dropna()
            s.index = pd.to_datetime(s.index)
            s = s.astype(float)
            return s
        except Exception as e:
            last_err = e
            wait = 2 * i
            print(f"[fredapi] attempt {i+1} failed for {code}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise last_err

def to_month_start_from_daily(s: pd.Series) -> pd.Series:
    """Daily -> month-end last -> set index to month start (YYYY-MM-01)."""
    if s.empty:
        return s
    s = s.sort_index()
    s = s.resample("ME").last()
    s.index = s.index.to_period("M").to_timestamp(how="start")
    return s.dropna()

def to_month_start_from_monthly(s: pd.Series) -> pd.Series:
    """Monthly series -> set index to month start (YYYY-MM-01)."""
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp(how="start")
    return s

def to_month_start_from_quarterly(s: pd.Series) -> pd.Series:
    """Quarterly -> quarter start -> reindex monthly (MS) with forward-fill."""
    if s.empty:
        return s
    s = s.sort_index()
    s.index = s.index.to_period("Q").to_timestamp(how="start")  # quarter start
    # monthly index from first to last, month starts
    m_idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(m_idx).ffill()
    return s

def yoy_pct(s: pd.Series) -> pd.Series:
    return (s / s.shift(12) - 1.0) * 100.0

def mom_pct(s: pd.Series) -> pd.Series:
    return (s / s.shift(1) - 1.0) * 100.0

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ===================== Main =====================

def main():
    # ---------- Pull macro ----------
    indpro_m = to_month_start_from_monthly(safe_get_series(INDPRO, START)).rename("indpro")
    lei_m = to_month_start_from_monthly(safe_get_series(USSLIND, START)).rename("lei")

    dgs10_m = to_month_start_from_daily(safe_get_series(DGS10, START)).rename("dgs10")
    dgs3mo_m = to_month_start_from_daily(safe_get_series(DGS3MO, START)).rename("dgs3mo")
    yield_curve_bps = ((dgs10_m - dgs3mo_m) * 100.0).rename("yield_curve_spread")

    # Core macro signals
    indpro_yoy = yoy_pct(indpro_m).rename("indpro_yoy")
    lei_yoy = yoy_pct(lei_m).rename("lei_yoy")

    # (optional) smooth YoY with 3m SMA to mimic your prior *_yoy_sm columns
    indpro_yoy_sm = indpro_yoy.rolling(3, min_periods=1).mean().rename("indpro_yoy_sm")
    lei_yoy_sm = lei_yoy.rolling(3, min_periods=1).mean().rename("lei_yoy_sm")

    indpro_signal = (indpro_yoy_sm > 0).astype(int).rename("indpro_signal")
    lei_signal = (lei_yoy_sm > 0).astype(int).rename("lei_signal")
    yield_curve_signal = (yield_curve_bps > 0).astype(int).rename("yield_curve_signal")

    macro_score = (indpro_signal + lei_signal + yield_curve_signal).rename("macro_score")
    macro_regime = macro_score.apply(lambda x: "expansion" if x >= 2 else "contraction").rename("macro_regime")

    # Additional macro indicators
    hy_oas_m = to_month_start_from_daily(safe_get_series(HY_OAS, START)).rename("hy_oas")      # bps
    unrate_m = to_month_start_from_monthly(safe_get_series(UNRATE, START)).rename("unrate")   # %
    cpi_m = to_month_start_from_monthly(safe_get_series(CPI, START)).rename("cpi")
    cpi_yoy = yoy_pct(cpi_m).rename("cpi_yoy")

    gdp_q = to_month_start_from_quarterly(safe_get_series(GDP_SAAR, START)).rename("gdp_saar")  # % (QoQ SAAR)
    umcsent_m = to_month_start_from_monthly(safe_get_series(UMCSENT, START)).rename("umcsent")
    houst_m = to_month_start_from_monthly(safe_get_series(HOUST, START)).rename("houst")

    sp500_m = to_month_start_from_daily(safe_get_series(SP500, START)).rename("sp500")

    # ---------- Risk (FRED-only) ----------
    vix_m = to_month_start_from_daily(safe_get_series(VIXCLS, START)).rename("vix")

    # MoM/YoY for risk levels (handy for UI)
    sp500_mom = mom_pct(sp500_m).rename("sp500_mom")
    sp500_yoy = yoy_pct(sp500_m).rename("sp500_yoy")

    hy_oas_mom = mom_pct(hy_oas_m).rename("hy_oas_mom")
    hy_oas_yoy = yoy_pct(hy_oas_m).rename("hy_oas_yoy")

    vix_mom = mom_pct(vix_m).rename("vix_mom")
    vix_yoy = yoy_pct(vix_m).rename("vix_yoy")

    # Trend signal: SP500 vs 10-month SMA
    sp500_sma_10m = sp500_m.rolling(TREND_SMA_MONTHS, min_periods=TREND_SMA_MONTHS).mean().rename("sp500_sma_10m")
    trend_on = (sp500_m >= sp500_sma_10m).astype(int).rename("trend_on")

    # Credit signal: HY OAS vs max(500 bps, 12-month SMA)
    hy_oas_sma_12m = hy_oas_m.rolling(CREDIT_SMA_MONTHS, min_periods=CREDIT_SMA_MONTHS).mean().rename("hy_oas_sma_12m")
    hy_barrier = pd.concat(
        [pd.Series(CREDIT_OAS_BPS_THRESHOLD, index=hy_oas_m.index), hy_oas_sma_12m], axis=1
    ).max(axis=1)
    credit_off = (hy_oas_m > hy_barrier).astype(int).rename("credit_off")

    # Vol signal: VIX > 25
    vix_off = (vix_m > VIX_THRESHOLD).astype(int).rename("vix_off")

    off_votes = (credit_off + vix_off).rename("off_votes")

    def classify_risk(row):
        if np.isnan(row.get("trend_on")) or np.isnan(row.get("off_votes")):
            return np.nan
        if row["trend_on"] == 1 and row["off_votes"] == 0:
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"

    # ---------- Assemble ----------
    df = (
        indpro_m.to_frame()
        .join([indpro_yoy_sm, indpro_signal], how="outer")
        .join([lei_m, lei_yoy_sm, lei_signal], how="outer")
        .join([dgs10_m, dgs3mo_m, yield_curve_bps, yield_curve_signal], how="outer")
        .join([macro_score, macro_regime], how="outer")
        .join([hy_oas_m, unrate_m, cpi_yoy, gdp_q, umcsent_m, houst_m], how="outer")
        .join([sp500_m], how="outer")
        .join([vix_m], how="outer")
        # risk extras
        .join([sp500_mom, sp500_yoy, sp500_sma_10m, trend_on], how="outer")
        .join([hy_oas_mom, hy_oas_yoy, hy_oas_sma_12m, credit_off], how="outer")
        .join([vix_mom, vix_yoy, vix_off, off_votes], how="outer")
        .sort_index()
    )

    df["risk_regime"] = df.apply(classify_risk, axis=1)

    # (Optional) keep rows where we have at least macro core (or risk trio)
    core_cols = ["indpro", "lei", "yield_curve_spread"]
    risk_cols = ["sp500", "hy_oas", "vix"]
    keep_mask = df[core_cols].notna().any(axis=1) | df[risk_cols].notna().any(axis=1)
    df = df[keep_mask]

    # Final formatting
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Order columns: your original header first, then the risk block
    cols_macro_first = [
        "date",
        "indpro", "indpro_yoy_sm", "indpro_signal",
        "lei", "lei_yoy_sm", "lei_signal",
        "dgs10", "dgs3mo", "yield_curve_spread", "yield_curve_signal",
        "macro_score", "macro_regime",
        "hy_oas", "unrate", "cpi_yoy", "gdp_saar", "umcsent", "houst",
        "sp500",
        # sp500_earnings_yield kept for compatibility (not derivable from FRED alone)
        # leave as NaN so downstream isn’t broken if it expects the column
        "sp500_earnings_yield",
    ]

    # Ensure the placeholder column exists
    if "sp500_earnings_yield" not in df.columns:
        df["sp500_earnings_yield"] = np.nan

    cols_risk_block = [
        # Risk levels + changes
        "sp500_mom", "sp500_yoy",
        "hy_oas_mom", "hy_oas_yoy",
        "vix", "vix_mom", "vix_yoy",
        # Risk SMAs + signals + composite
        "sp500_sma_10m", "trend_on",
        "hy_oas_sma_12m", "credit_off",
        "vix_off", "off_votes", "risk_regime",
    ]

    # Build final ordered list (only include columns that exist)
    final_cols = [c for c in cols_macro_first if c in df.columns] + [c for c in cols_risk_block if c in df.columns]
    df = df[final_cols]

    # Append + de-duplicate
    ensure_dir(OUT_PATH)
    if os.path.exists(OUT_PATH):
        try:
            old = pd.read_csv(OUT_PATH, parse_dates=["date"])
            old["date"] = old["date"].dt.strftime("%Y-%m-%d")
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            combined.to_csv(OUT_PATH, index=False)
            print(f"Wrote {OUT_PATH} with {len(combined):,} rows (latest {combined['date'].iloc[-1]})")
            return
        except Exception as e:
            print(f"[WARN] Could not merge with existing {OUT_PATH}: {e}", file=sys.stderr)

    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(df):,} rows (latest {df['date'].iloc[-1]})")


if __name__ == "__main__":
    pd.set_option("display.width", 180)
    pd.set_option("display.max_columns", 200)
    main()