# src/compute_macro_monthly.py
# Unified Macro + Risk monthly pipeline (FRED-only) with per-series completion rules.
# - Daily series: monthly median; if a month missing entirely -> time interpolate, then ffill/bfill.
# - Monthly series: use monthly; if missing -> time interpolate (avg of prior & next), then ffill/bfill at edges.
# - Quarterly series: repeat value to all months in quarter (ffill monthly), then bfill at start.
# - MoM: daily/monthly -> pct_change(1); quarterly -> pct_change(3).
# - YoY: pct_change(12).
# - Macro signals: INDPRO YoY>0, USSLIND YoY>0, spread(DGS10-DGS3MO)>0 (bps)
# - Risk signals: SP500 >= 10m SMA -> trend_on; HY OAS > max(500, 12m SMA) -> credit_off; VIX > 25 -> vix_off
# - Writes/merges: data/macro_monthly_history.csv

import os
import sys
import time
import math
import pandas as pd
from datetime import datetime
from fredapi import Fred

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY is not set in the environment.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

OUT_PATH   = "data/macro_monthly_history.csv"
START_DATE = "1990-01-01"

# ------------ Series codes (FRED) ------------
# Macro Core 3
INDPRO   = "INDPRO"              # monthly (idx)
USSLIND  = "USSLIND"             # monthly (idx)
DGS10    = "DGS10"               # daily (%)
DGS3MO   = "DGS3MO"              # daily (%)

# Additional Macro
BAML_OAS = "BAMLH0A0HYM2"        # daily (bps)
UNRATE   = "UNRATE"              # monthly (%)
CPI      = "CPIAUCSL"            # monthly (idx)
GDP_SAAR = "A191RL1Q225SBEA"     # quarterly (%, QoQ SAAR)
UMCSENT  = "UMCSENT"             # monthly (idx)
HOUST    = "HOUST"               # monthly (units, SAAR)

# Risk (from FRED)
SP500    = "SP500"               # daily (index level)
VIXCLS   = "VIXCLS"              # daily (level)

# ------------ Risk parameters ------------
TREND_SMA_MONTHS = 10
CREDIT_SMA_MONTHS = 12
CREDIT_OAS_BPS_THRESHOLD = 500.0
VIX_THRESHOLD = 25.0

# ------------ Helpers ------------
def month_index(start=START_DATE, end=None):
    if end is None:
        # up to current month start
        end = pd.Timestamp.today().to_period("M").to_timestamp("MS")
    idx = pd.date_range(pd.to_datetime(start).to_period("M").to_timestamp("MS"),
                        end, freq="MS")
    return idx

def safe_fred(code, start=START_DATE):
    last_err = None
    for i in range(3):
        try:
            s = fred.get_series(code, observation_start=start)
            s = pd.Series(s)
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            return s.dropna()
        except Exception as e:
            last_err = e
            wait = 2 * i
            print(f"[fredapi] attempt {i+1} failed for {code}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise last_err

def complete_daily_to_monthly_median(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Daily -> monthly median, aligned to month-start. Fill any empty months via time interpolation, then ffill/bfill."""
    # Resample to monthly start using median of daily obs within the month
    m = s.resample("MS").median()  # median of all daily values per calendar month
    # Align to full index
    m = m.reindex(idx)
    # Fill gaps via time-based interpolation (avg of neighbors when single missing)
    m = m.interpolate(method="time", limit_direction="both")
    # Final pad in case edges remain
    m = m.ffill().bfill()
    return m

def complete_monthly(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Monthly source -> align; fill missing as average of prior/next (time interpolation), then ffill/bfill edges."""
    # Ensure monthly start index for the raw series
    s_m = s.resample("MS").last()
    s_m = s_m.reindex(idx)
    s_m = s_m.interpolate(method="time", limit_direction="both")
    s_m = s_m.ffill().bfill()
    return s_m

def complete_quarterly_to_monthly(s: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    """Quarterly source -> repeat the quarter's value for each month in the quarter (ffill), then bfill edges."""
    # Put quarterly points onto monthly grid by forward fill
    # First, ensure the series has a monthly index holding quarter-end points
    s_q = s.copy()
    s_q = s_q.resample("M").last()  # quarter prints typically at quarter-end month
    s_q = s_q.reindex(pd.date_range(idx.min(), idx.max(), freq="M"))
    s_q = s_q.ffill().bfill()
    # Now place onto month-start index
    s_ms = s_q.resample("MS").first()  # convert "month-end aligned" to month-start stamps
    s_ms = s_ms.reindex(idx)
    s_ms = s_ms.ffill().bfill()
    return s_ms

def pct_mom(series: pd.Series, quarterly: bool) -> pd.Series:
    """MoM: daily/monthly -> 1 month; quarterly -> 3 months (month vs prior quarter)."""
    periods = 3 if quarterly else 1
    return series.pct_change(periods) * 100.0

def pct_yoy(series: pd.Series) -> pd.Series:
    return series.pct_change(12) * 100.0

# ------------ Main ------------
def main():
    idx = month_index()

    # ---- Pull raw ----
    s_indpro   = safe_fred(INDPRO, START_DATE)
    s_lei      = safe_fred(USSLIND, START_DATE)
    s_dgs10    = safe_fred(DGS10, START_DATE)
    s_dgs3mo   = safe_fred(DGS3MO, START_DATE)
    s_hyoas    = safe_fred(BAML_OAS, START_DATE)
    s_unrate   = safe_fred(UNRATE, START_DATE)
    s_cpi      = safe_fred(CPI, START_DATE)
    s_gdp      = safe_fred(GDP_SAAR, START_DATE)
    s_umcsent  = safe_fred(UMCSENT, START_DATE)
    s_houst    = safe_fred(HOUST, START_DATE)
    s_sp500    = safe_fred(SP500, START_DATE)
    s_vix      = safe_fred(VIXCLS, START_DATE)

    # ---- Complete to monthly per your rules ----
    # Daily -> monthly median
    dgs10_m   = complete_daily_to_monthly_median(s_dgs10, idx)
    dgs3mo_m  = complete_daily_to_monthly_median(s_dgs3mo, idx)
    hyoas_m   = complete_daily_to_monthly_median(s_hyoas, idx)
    sp500_m   = complete_daily_to_monthly_median(s_sp500, idx)
    vix_m     = complete_daily_to_monthly_median(s_vix, idx)

    # Monthly -> monthly with time interpolation for gaps
    indpro_m  = complete_monthly(s_indpro, idx)
    lei_m     = complete_monthly(s_lei, idx)
    unrate_m  = complete_monthly(s_unrate, idx)
    cpi_m     = complete_monthly(s_cpi, idx)
    umcsent_m = complete_monthly(s_umcsent, idx)
    houst_m   = complete_monthly(s_houst, idx)

    # Quarterly -> monthly repeat per quarter
    gdp_m     = complete_quarterly_to_monthly(s_gdp, idx)

    # ---- Derived: yield curve spread (bps) ----
    spread_m = (dgs10_m - dgs3mo_m) * 100.0

    # ---- MoM & YoY for each metric ----
    # Quarterly flag only for GDP
    metrics = {
        "indpro":      (indpro_m,      False),
        "lei":         (lei_m,         False),
        "yield_curve": (spread_m,      False),
        "hy_oas":      (hyoas_m,       False),
        "unrate":      (unrate_m,      False),
        "cpi":         (cpi_m,         False),
        "gdp_saar":    (gdp_m,         True),   # quarterly logic
        "umcsent":     (umcsent_m,     False),
        "houst":       (houst_m,       False),
        "sp500":       (sp500_m,       False),
        "vix":         (vix_m,         False),
        "dgs10":       (dgs10_m,       False),
        "dgs3mo":      (dgs3mo_m,      False),
    }

    frames = []
    for name, (ser, is_quarterly) in metrics.items():
        df = pd.DataFrame({name: ser})
        df[f"{name}_mom"] = pct_mom(ser, quarterly=is_quarterly)
        df[f"{name}_yoy"] = pct_yoy(ser)
        frames.append(df)

    all_df = pd.concat(frames, axis=1).reindex(idx).sort_index()

    # ---- Macro signals/scores ----
    # YoY for indpro/lei computed above
    indpro_sig = (all_df["indpro_yoy"] > 0).astype(int)
    lei_sig    = (all_df["lei_yoy"] > 0).astype(int)
    curve_sig  = (all_df["yield_curve"] > 0).astype(int)

    macro_score  = (indpro_sig + lei_sig + curve_sig).astype(int)
    macro_regime = macro_score.apply(lambda x: "expansion" if x >= 2 else "contraction")

    all_df["indpro_signal"]        = indpro_sig
    all_df["lei_signal"]           = lei_sig
    all_df["yield_curve_signal"]   = curve_sig
    all_df["macro_score"]          = macro_score
    all_df["macro_regime"]         = macro_regime

    # ---- Risk signals/scores ----
    # SP500 10m SMA (monthly series), HY OAS 12m SMA (monthly series)
    all_df["sp500_sma_10m"]  = all_df["sp500"].rolling(TREND_SMA_MONTHS, min_periods=TREND_SMA_MONTHS).mean()
    all_df["hy_oas_sma_12m"] = all_df["hy_oas"].rolling(CREDIT_SMA_MONTHS, min_periods=CREDIT_SMA_MONTHS).mean()

    all_df["trend_on"] = (all_df["sp500"] >= all_df["sp500_sma_10m"]).astype(int)

    hy_barrier = pd.concat(
        [
            pd.Series(CREDIT_OAS_BPS_THRESHOLD, index=all_df.index),
            all_df["hy_oas_sma_12m"]
        ],
        axis=1
    ).max(axis=1)
    all_df["credit_off"] = (all_df["hy_oas"] > hy_barrier).astype(int)

    all_df["vix_off"] = (all_df["vix"] > VIX_THRESHOLD).astype(int)

    all_df["off_votes"] = (all_df["credit_off"] + all_df["vix_off"]).astype(int)

    def risk_reg(row):
        if (row["trend_on"] == 1) and (row["off_votes"] == 0):
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"

    all_df["risk_regime"] = all_df.apply(risk_reg, axis=1)

    # ---- Final formatting ----
    out = all_df.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    # Column order (grouped)
    ordered = [
        "date",
        # Core macro levels
        "indpro","lei","yield_curve",
        "indpro_mom","lei_mom","yield_curve_mom",
        "indpro_yoy","lei_yoy","yield_curve_yoy",
        "indpro_signal","lei_signal","yield_curve_signal","macro_score","macro_regime",
        # Additional macro levels
        "hy_oas","unrate","cpi","gdp_saar","umcsent","houst",
        "hy_oas_mom","unrate_mom","cpi_mom","gdp_saar_mom","umcsent_mom","houst_mom",
        "hy_oas_yoy","unrate_yoy","cpi_yoy","gdp_saar_yoy","umcsent_yoy","houst_yoy",
        # Risk inputs
        "sp500","sp500_sma_10m","vix","hy_oas_sma_12m",
        "sp500_mom","vix_mom","hy_oas_mom",
        "sp500_yoy","vix_yoy","hy_oas_yoy",
        # Risk signals
        "trend_on","credit_off","vix_off","off_votes","risk_regime",
        # Raw rates levels (optional reference)
        "dgs10","dgs3mo","dgs10_mom","dgs3mo_mom","dgs10_yoy","dgs3mo_yoy",
    ]
    # Keep only columns that exist
    cols = [c for c in ordered if c in out.columns]
    out = out[cols]

    # ---- Save (append + de-dup on date) ----
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        try:
            old = pd.read_csv(OUT_PATH, parse_dates=["date"])
            old["date"] = old["date"].dt.strftime("%Y-%m-%d")
            combined = pd.concat([old, out], ignore_index=True)
            combined = combined.sort_values("date").drop_duplicates(subset=["date"], keep="last")
            combined.to_csv(OUT_PATH, index=False)
        except Exception as e:
            print(f"[WARN] Could not merge with existing {OUT_PATH}: {e}", file=sys.stderr)
            out.to_csv(OUT_PATH, index=False)
    else:
        out.to_csv(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH} with {len(out)} rows (latest {out['date'].iloc[-1]})")

if __name__ == "__main__":
    main()