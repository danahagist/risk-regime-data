# src/compute_macro_monthly.py
# Combined Macro + Risk (FRED-only), 2020+ with interpolation stats.
# - Daily series -> monthly median (month-start date)
# - Monthly series -> neighbor-avg fill for isolated missing months (per-series only)
# - Quarterly series -> repeat across months in quarter
# - Computes MoM% & YoY% for all series
# - Macro core signals: INDPRO YoY>0, USSLIND YoY>0, (DGS10-DGS3MO)>0
# - Risk signals: Trend (SP500 >= 10m SMA), Credit (HY OAS > max(500, 12m SMA)), Vol (VIX>25)
# - Outputs:
#     data/macro_monthly_history.csv
#     data/interpolation_stats.csv

import os
import sys
import time
import pandas as pd
import numpy as np
from fredapi import Fred

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY not set.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

# --------------------
# Config
# --------------------
START_DATE = "2020-01-01"  # restrict to 2020+
OUT_PATH    = "data/macro_monthly_history.csv"
STATS_PATH  = "data/interpolation_stats.csv"

# FRED codes
INDPRO   = "INDPRO"            # Industrial Production (monthly)
USSLIND  = "USSLIND"           # Leading Index (monthly)
DGS10    = "DGS10"             # 10Y Treasury (daily)
DGS3MO   = "DGS3MO"            # 3M T-Bill (daily)
BAML_HYOAS = "BAMLH0A0HYM2"    # HY OAS (daily)
UNRATE   = "UNRATE"            # Unemployment (monthly)
CPI      = "CPIAUCSL"          # CPI (monthly)
GDP_SAAR = "A191RL1Q225SBEA"   # Real GDP QoQ SAAR (quarterly)
UMCSENT  = "UMCSENT"           # Michigan Sentiment (monthly)
HOUST    = "HOUST"             # Housing Starts (monthly)
SP500    = "SP500"             # S&P 500 Level (daily)
VIXCLS   = "VIXCLS"            # VIX (daily)

# Risk thresholds/params
TREND_SMA_MONTHS  = 10
CREDIT_SMA_MONTHS = 12
CREDIT_OAS_FLOOR  = 500.0
VIX_THRESHOLD     = 25.0

# --------------------
# Helpers
# --------------------
def month_index():
    start = pd.Timestamp(START_DATE).replace(day=1)
    end = pd.Timestamp.today().replace(day=1)
    return pd.date_range(start, end, freq="MS")

def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def fred_series(code: str, start: str) -> pd.Series:
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

def to_month_start_daily_median(s: pd.Series, idx: pd.DatetimeIndex):
    """
    Daily -> (series, stats)
    - monthly median stamped to month-start (idx)
    - stats: months_with_data, months_without_data
    """
    if s.empty:
        ser = pd.Series(index=idx, dtype="float64")
        return ser, {"metric": "", "method":"daily_median", "months_with_data":0, "months_without_data":len(idx), "filled_count":0, "filled_pct":0.0}
    s = s.sort_index()
    # compute median per calendar month on an MS index
    med = s.resample("MS").median()
    ser = med.reindex(idx)

    months_with_data = ser.notna().sum()
    months_without_data = ser.isna().sum()
    # we do NOT interpolate daily medians across empty months
    stats = {
        "metric": "",  # fill later
        "method": "daily_median",
        "months_with_data": int(months_with_data),
        "months_without_data": int(months_without_data),
        "filled_count": 0,          # we didn't fill across months here
        "filled_pct": 0.0
    }
    return ser, stats

def to_monthly_with_neighbor_avg_fill(s: pd.Series, idx: pd.DatetimeIndex):
    """
    Monthly -> (series, stats)
    - normalize to MS, align to idx
    - fill only single-month interior gaps with (prev+next)/2, per-series (no cross-series effects)
    - stats returns how many fills happened
    """
    s = s.copy().sort_index()
    s.index = s.index.to_period("M").to_timestamp()
    s = s.groupby(level=0).last()
    ser = s.reindex(idx)

    before_na = ser.isna().sum()
    # only fill where both neighbors exist
    mask = ser.isna()
    prev = ser.shift(1)
    nxt  = ser.shift(-1)
    fill_candidates = mask & prev.notna() & nxt.notna()
    ser.loc[fill_candidates] = (prev + nxt) / 2.0

    filled_count = int(fill_candidates.sum())
    after_na = ser.isna().sum()
    # any remaining NaNs are left as-is
    stats = {
        "metric": "",  # fill later
        "method": "monthly_neighbor_avg",
        "months_with_data": int(ser.notna().sum()),
        "months_without_data": int(ser.isna().sum()),
        "filled_count": filled_count,
        "filled_pct": float(filled_count) / float(len(idx)) * 100.0 if len(idx) else 0.0,
    }
    return ser, stats

def quarterly_to_months_repeat(s: pd.Series, idx: pd.DatetimeIndex):
    """
    Quarterly -> (series, stats)
    - normalize to quarter start, forward-fill to months
    - all months within a quarter are repeats; reported as 'repeated_count' (informational)
    """
    if s.empty:
        ser = pd.Series(index=idx, dtype="float64")
        stats = {
            "metric": "",
            "method": "quarterly_repeat",
            "months_with_data": 0,
            "months_without_data": len(idx),
            "filled_count": 0,      # not interpolation; just expansion
            "filled_pct": 0.0,
            "repeated_count": 0
        }
        return ser, stats

    s = s.copy().sort_index()
    s.index = s.index.to_period("Q").to_timestamp(how="S")
    m = s.reindex(pd.date_range(idx.min(), idx.max(), freq="MS")).ffill()
    ser = m.reindex(idx)

    # Count how many months are from repeats (all but first month of each quarter after an observation)
    # This is informational; not counted as "filled/interpolated".
    q_labels = ser.index.to_period("Q")
    # months that are not the first month of their quarter are repeats if value equals previous month
    repeated = (ser == ser.shift(1)) & (q_labels == q_labels.shift(1)) & (ser.notna())
    repeated_count = int(repeated.sum())

    stats = {
        "metric": "",
        "method": "quarterly_repeat",
        "months_with_data": int(ser.notna().sum()),
        "months_without_data": int(ser.isna().sum()),
        "filled_count": 0,   # we don't treat repeat as interpolation
        "filled_pct": 0.0,
        "repeated_count": repeated_count
    }
    return ser, stats

def pct_mom(series: pd.Series) -> pd.Series:
    return series.astype(float).pct_change(1) * 100.0

def pct_yoy(series: pd.Series) -> pd.Series:
    return series.astype(float).pct_change(12) * 100.0

# --------------------
# Main
# --------------------
def main():
    idx = month_index()
    stats_rows = []

    # ---- Pull & transform
    # Monthly with neighbor-avg fill
    indpro_s, st = to_monthly_with_neighbor_avg_fill(fred_series(INDPRO,   START_DATE), idx); st["metric"]="indpro";  stats_rows.append(st)
    usslind_s, st= to_monthly_with_neighbor_avg_fill(fred_series(USSLIND,  START_DATE), idx); st["metric"]="lei";     stats_rows.append(st)
    unrate_s, st = to_monthly_with_neighbor_avg_fill(fred_series(UNRATE,   START_DATE), idx); st["metric"]="unrate";  stats_rows.append(st)
    cpi_s, st    = to_monthly_with_neighbor_avg_fill(fred_series(CPI,      START_DATE), idx); st["metric"]="cpi";     stats_rows.append(st)
    umcsent_s, st= to_monthly_with_neighbor_avg_fill(fred_series(UMCSENT,  START_DATE), idx); st["metric"]="umcsent"; stats_rows.append(st)
    houst_s, st  = to_monthly_with_neighbor_avg_fill(fred_series(HOUST,    START_DATE), idx); st["metric"]="houst";   stats_rows.append(st)

    # Quarterly -> months
    gdp_s, st    = quarterly_to_months_repeat(fred_series(GDP_SAAR, START_DATE), idx); st["metric"]="gdp_saar"; stats_rows.append(st)

    # Daily -> monthly median
    dgs10_d = fred_series(DGS10,  START_DATE)
    dgs3m_d = fred_series(DGS3MO, START_DATE)
    hy_d    = fred_series(BAML_HYOAS, START_DATE)
    spx_d   = fred_series(SP500,  START_DATE)
    vix_d   = fred_series(VIXCLS, START_DATE)

    dgs10_s, st = to_month_start_daily_median(dgs10_d, idx); st["metric"]="dgs10";  stats_rows.append(st)
    dgs3m_s, st = to_month_start_daily_median(dgs3m_d, idx); st["metric"]="dgs3mo"; stats_rows.append(st)
    hy_s,    st = to_month_start_daily_median(hy_d,    idx); st["metric"]="hy_oas"; stats_rows.append(st)
    spx_s,   st = to_month_start_daily_median(spx_d,   idx); st["metric"]="sp500";  stats_rows.append(st)
    vix_s,   st = to_month_start_daily_median(vix_d,   idx); st["metric"]="vix";    stats_rows.append(st)

    # Yield curve spread (in percentage points; switch to *100 for bps if desired)
    yc_spread = (dgs10_s - dgs3m_s).rename("yield_curve_spread")

    # ---- Assemble core DataFrame (aligned to idx)
    core = pd.concat(
        [indpro_s.rename("indpro"),
         usslind_s.rename("lei"),
         dgs10_s.rename("dgs10"),
         dgs3m_s.rename("dgs3mo"),
         yc_spread,
         hy_s.rename("hy_oas"),
         unrate_s.rename("unrate"),
         cpi_s.rename("cpi"),
         gdp_s.rename("gdp_saar"),
         umcsent_s.rename("umcsent"),
         houst_s.rename("houst"),
         spx_s.rename("sp500"),
         vix_s.rename("vix")],
        axis=1
    ).reindex(idx)

    # ---- Changes (MoM / YoY) for each series
    def add_changes(df, col):
        df[f"{col}_mom"] = pct_mom(df[col])
        df[f"{col}_yoy"] = pct_yoy(df[col])

    for col in core.columns:
        add_changes(core, col)

    # ---- Macro core signals
    core["indpro_signal"] = (core["indpro_yoy"] > 0).astype(int)
    core["lei_signal"] = (core["lei_yoy"] > 0).astype(int)
    core["yield_curve_signal"] = (core["yield_curve_spread"] > 0).astype(int)
    core["macro_score"] = core["indpro_signal"] + core["lei_signal"] + core["yield_curve_signal"]
    core["macro_regime"] = np.where(core["macro_score"] >= 2, "expansion", "contraction")

    # ---- Risk signals (monthly)
    core["sp500_sma_10m"] = core["sp500"].rolling(TREND_SMA_MONTHS, min_periods=TREND_SMA_MONTHS).mean()
    core["trend_on"] = (core["sp500"] >= core["sp500_sma_10m"]).astype(int)

    core["hy_oas_sma_12m"] = core["hy_oas"].rolling(CREDIT_SMA_MONTHS, min_periods=CREDIT_SMA_MONTHS).mean()
    credit_barrier = pd.concat(
        [pd.Series(CREDIT_OAS_FLOOR, index=core.index), core["hy_oas_sma_12m"]],
        axis=1
    ).max(axis=1)
    core["credit_off"] = (core["hy_oas"] > credit_barrier).astype(int)

    core["vix_off"] = (core["vix"] > VIX_THRESHOLD).astype(int)
    core["off_votes"] = core["credit_off"] + core["vix_off"]
    core["risk_regime"] = np.where(
        (core["trend_on"] == 1) & (core["off_votes"] == 0), "On",
        np.where(core["off_votes"] >= 2, "Off", "Mixed")
    )

    # ---- Final formatting & column order (regimes/scores first)
    core = core.reset_index().rename(columns={"index": "date"})
    core["date"] = core["date"].dt.strftime("%Y-%m-%d")

    FRONT = [
        "date",
        # Risk first
        "trend_on", "credit_off", "vix_off", "off_votes", "risk_regime",
        # Macro core
        "indpro_signal", "lei_signal", "yield_curve_signal", "macro_score", "macro_regime",
    ]
    RISK_BLOCK = [
        "sp500", "sp500_mom", "sp500_yoy", "sp500_sma_10m",
        "hy_oas", "hy_oas_mom", "hy_oas_yoy", "hy_oas_sma_12m",
        "vix", "vix_mom", "vix_yoy",
    ]
    MACRO_BLOCK = [
        "indpro", "indpro_mom", "indpro_yoy",
        "lei", "lei_mom", "lei_yoy",
        "dgs10", "dgs10_mom", "dgs10_yoy",
        "dgs3mo", "dgs3mo_mom", "dgs3mo_yoy",
        "yield_curve_spread", "yield_curve_spread_mom", "yield_curve_spread_yoy",
        "unrate", "unrate_mom", "unrate_yoy",
        "cpi", "cpi_mom", "cpi_yoy",
        "gdp_saar", "gdp_saar_mom", "gdp_saar_yoy",
        "umcsent", "umcsent_mom", "umcsent_yoy",
        "houst", "houst_mom", "houst_yoy",
    ]

    # Keep only columns that exist (in case early periods lack SMA windows)
    ordered_cols = [c for c in FRONT + RISK_BLOCK + MACRO_BLOCK if c in core.columns]
    core = core[ordered_cols]

    # ---- Write main CSV (append + dedupe by date)
    ensure_dir(OUT_PATH)
    if os.path.exists(OUT_PATH):
        try:
            old = pd.read_csv(OUT_PATH, dtype=str)
            combined = pd.concat([old, core], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date")
            combined.to_csv(OUT_PATH, index=False)
        except Exception as e:
            print(f"[WARN] Could not merge with existing {OUT_PATH}: {e}", file=sys.stderr)
            core.to_csv(OUT_PATH, index=False)
    else:
        core.to_csv(OUT_PATH, index=False)

    # ---- Build & write interpolation stats CSV
    stats_df = pd.DataFrame(stats_rows)
    # Add totals & percents normalized to index length
    total_months = len(idx)
    stats_df["total_months"] = total_months
    # Ensure consistent column order
    cols = [
        "metric", "method",
        "months_with_data", "months_without_data",
        "filled_count", "filled_pct",
        "repeated_count",  # present for quarterly; NaN for others
        "total_months"
    ]
    for c in cols:
        if c not in stats_df.columns:
            stats_df[c] = np.nan
    ensure_dir(STATS_PATH)
    stats_df = stats_df[cols].sort_values(["metric", "method"])
    stats_df.to_csv(STATS_PATH, index=False)

    print(f"Wrote {OUT_PATH} and {STATS_PATH}. Latest date: {core['date'].iloc[-1]}")

if __name__ == "__main__":
    main()