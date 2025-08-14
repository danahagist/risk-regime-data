import os
import sys
import time
import math
import pandas as pd
import numpy as np
from fredapi import Fred

# ------------------------
# Config
# ------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY is not set.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

OUT_PATH = "data/macro_monthly_history.csv"
INTERP_REPORT_PATH = "data/interpolation_report.csv"
START_DATE = "2020-01-01"   # limit as requested

# FRED codes
INDPRO = "INDPRO"                  # Industrial Production (index)
CFNAI  = "CFNAI"                   # Chicago Fed National Activity Index
DGS10  = "DGS10"                   # 10Y Treasury (percent, daily)
DGS3MO = "DGS3MO"                  # 3M Treasury (percent, daily)
BAML_OAS = "BAMLH0A0HYM2"          # HY OAS (bps, daily)
UNRATE = "UNRATE"                  # Unemployment rate (%)
CPI    = "CPIAUCSL"                # CPI (index, SA)
GDP_Q  = "A191RL1Q225SBEA"         # Real GDP QoQ SAAR (%), quarterly
UMCSENT = "UMCSENT"                # Michigan Sentiment (index)
HOUST   = "HOUST"                  # Housing starts (SAAR)
VIXCLS  = "VIXCLS"                 # VIX (index level, daily)
SP500   = "SP500"                  # S&P 500 (index level, daily)

# Risk params
TREND_SMA_MONTHS = 10
CREDIT_SMA_MONTHS = 12
CREDIT_OAS_BPS_THRESHOLD = 500.0
VIX_THRESHOLD = 25.0

# ------------------------
# Helpers
# ------------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def month_index(start=START_DATE, end=None):
    if end is None:
        end = pd.Timestamp.today().to_period("M").to_timestamp(how="start")
    idx = pd.date_range(start=pd.Timestamp(start), end=end, freq="MS")  # 1st of month
    return idx

def fred_series(code: str, start: str) -> pd.Series:
    """Fetch FRED series with simple retries; return DatetimeIndex Series."""
    last_err = None
    for i in range(4):
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

def to_monthly_from_daily(s: pd.Series) -> pd.Series:
    """Daily -> monthly median; index to month-start; independent interpolation (linear + edge fill)."""
    if s.empty:
        return s
    s = s.sort_index()
    m = s.resample("M").median()  # month-end labels
    # relabel to month start (avoid 'MS' freq string in Period)
    m.index = m.index.to_period("M").to_timestamp(how="start")
    # ensure full monthly index, then interpolate within the series (no cross-metric impact)
    full_idx = month_index(start=max(pd.Timestamp(START_DATE), m.index.min()))
    m = m.reindex(full_idx)
    interp_mask_before = m.isna()
    m = m.interpolate(method="linear", limit_direction="both")
    # for any leading/trailing NaNs still left (unlikely), ffill/bfill to ensure a value every month
    m = m.ffill().bfill()
    interp_mask_after = m.isna()
    # return series + a boolean mask of which points were originally interpolated
    return m, interp_mask_before & (~interp_mask_after)

def to_monthly_from_monthly(s: pd.Series) -> pd.Series:
    """Monthly -> monthly; fill missing by averaging prior/next (linear); edge ffill/bfill."""
    if s.empty:
        return s
    s = s.sort_index()
    s.index = s.index.to_period("M").to_timestamp(how="start")
    full_idx = month_index(start=max(pd.Timestamp(START_DATE), s.index.min()))
    m = s.reindex(full_idx)
    interp_mask_before = m.isna()
    m = m.interpolate(method="linear", limit_direction="both")
    m = m.ffill().bfill()
    interp_mask_after = m.isna()
    return m, interp_mask_before & (~interp_mask_after)

def to_monthly_from_quarterly(s: pd.Series) -> pd.Series:
    """Quarterly -> repeat same value for each month of the quarter; fill edges ffill/bfill."""
    if s.empty:
        return s
    s = s.sort_index()
    # make quarterly period index, then reindex to monthly (start-of-month) and ffill within quarter
    q = s.copy()
    # most FRED quarterly are end-of-quarter dates; convert robustly:
    q.index = q.index.to_period("Q").to_timestamp(how="end")
    monthly_idx = month_index(start=max(pd.Timestamp(START_DATE), q.index.min().to_period("M").to_timestamp(how="start")))
    # map each month to its quarter's value by forward-filling a monthly view
    qm = q.resample("MS").ffill()  # month-start freq, ffill inside quarter
    qm = qm.reindex(monthly_idx).ffill().bfill()
    # track "fills": months that did not align exactly to the original quarter endpoints
    # we’ll mark all months that are not at the original quarter-end month as "filled"
    quarter_end_ms = q.index.to_period("Q").to_timestamp(how="end").to_period("M").to_timestamp(how="start")
    filled_mask = ~qm.index.isin(quarter_end_ms)
    return qm, pd.Series(filled_mask, index=qm.index)

def mom_yoy_pct_monthly(level: pd.Series) -> (pd.Series, pd.Series):
    """MoM% and YoY% from a monthly level series."""
    x = level.astype(float)
    return x.pct_change(1) * 100.0, x.pct_change(12) * 100.0

def mom_yoy_pct_quarterly_month_view(level_q_monthly: pd.Series) -> (pd.Series, pd.Series):
    """
    For the monthly view of a quarterly series (repeated per month):
    - MoM%: month vs prior quarter’s value -> compute QoQ at quarter-ends, repeat across the quarter.
    - YoY%: vs same month last year (equivalently last year's quarter since repeated).
    """
    # identify quarter-end months
    idx = level_q_monthly.index
    q_end = (idx.to_period("Q").asfreq("M", how="end") == idx.to_period("M"))
    q_vals = level_q_monthly[q_end]
    q_qoq = q_vals.pct_change(1) * 100.0
    q_yoy = q_vals.pct_change(4) * 100.0
    # repeat each quarter's change across its 3 months
    mom = level_q_monthly.copy() * np.nan
    yoy = level_q_monthly.copy() * np.nan
    for t in q_vals.index:
        q_end_ts = t
        q_start_ts = (t - pd.offsets.MonthEnd(2)).to_period("M").to_timestamp(how="start")
        this_q = pd.date_range(q_start_ts, q_end_ts, freq="MS")
        mom.loc[this_q] = q_qoq.loc[t] if t in q_qoq.index else np.nan
        yoy.loc[this_q] = q_yoy.loc[t] if t in q_yoy.index else np.nan
    # edge fill to avoid NaN in early months where change not defined
    mom = mom.ffill().bfill()
    yoy = yoy.ffill().bfill()
    return mom, yoy

def rolling_sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=n).mean()

# ------------------------
# Main
# ------------------------
def main():
    ensure_dir(OUT_PATH)
    ensure_dir(INTERP_REPORT_PATH)

    # Build common monthly index through current month
    idx = month_index()

    # ---- Pull raw series
    indpro_raw = fred_series(INDPRO, START_DATE)        # monthly
    cfnai_raw  = fred_series(CFNAI, START_DATE)         # monthly
    dgs10_raw  = fred_series(DGS10, START_DATE)         # daily
    dgs3m_raw  = fred_series(DGS3MO, START_DATE)        # daily
    hy_oas_raw = fred_series(BAML_OAS, START_DATE)      # daily
    unrate_raw = fred_series(UNRATE, START_DATE)        # monthly
    cpi_raw    = fred_series(CPI, START_DATE)           # monthly
    gdp_raw    = fred_series(GDP_Q, START_DATE)         # quarterly (% SAAR)
    umcsent_raw= fred_series(UMCSENT, START_DATE)       # monthly
    houst_raw  = fred_series(HOUST, START_DATE)         # monthly
    vix_raw    = fred_series(VIXCLS, START_DATE)        # daily
    spx_raw    = fred_series(SP500, START_DATE)         # daily

    # ---- Transform to monthly series per policy, tracking interpolation masks
    indpro, indpro_interp = to_monthly_from_monthly(indpro_raw)
    cfnai,  cfnai_interp  = to_monthly_from_monthly(cfnai_raw)
    dgs10,  dgs10_interp  = to_monthly_from_daily(dgs10_raw)
    dgs3m,  dgs3m_interp  = to_monthly_from_daily(dgs3m_raw)
    hy_oas, hy_interp     = to_monthly_from_daily(hy_oas_raw)
    unrate, unrate_interp = to_monthly_from_monthly(unrate_raw)
    cpi,    cpi_interp    = to_monthly_from_monthly(cpi_raw)
    gdp_m,  gdp_interp    = to_monthly_from_quarterly(gdp_raw)  # repeated per month
    umcs,   umcs_interp   = to_monthly_from_monthly(umcsent_raw)
    houst,  houst_interp  = to_monthly_from_monthly(houst_raw)
    vix,    vix_interp    = to_monthly_from_daily(vix_raw)
    spx,    spx_interp    = to_monthly_from_daily(spx_raw)

    # Conform all series to common index (independent filling already done)
    series_dict = {
        "indpro": indpro.reindex(idx).ffill().bfill(),
        "cfnai":  cfnai.reindex(idx).ffill().bfill(),
        "dgs10":  dgs10.reindex(idx).ffill().bfill(),
        "dgs3mo": dgs3m.reindex(idx).ffill().bfill(),
        "hy_oas": hy_oas.reindex(idx).ffill().bfill(),
        "unrate": unrate.reindex(idx).ffill().bfill(),
        "cpi":    cpi.reindex(idx).ffill().bfill(),
        "gdp_saar": gdp_m.reindex(idx).ffill().bfill(),
        "umcsent": umcs.reindex(idx).ffill().bfill(),
        "houst":   houst.reindex(idx).ffill().bfill(),
        "vix":     vix.reindex(idx).ffill().bfill(),
        "sp500":   spx.reindex(idx).ffill().bfill(),
    }

    df = pd.DataFrame(series_dict, index=idx)

    # ---- Macro sub-calcs
    # Industrial Production YoY
    df["indpro_yoy"] = df["indpro"].pct_change(12) * 100.0

    # Yield curve (10Y – 3M) in bps
    df["yield_curve_spread"] = (df["dgs10"] - df["dgs3mo"]) * 100.0

    # CPI YoY (%)
    df["cpi_yoy"] = df["cpi"].pct_change(12) * 100.0

    # MoM/YoY for all metrics (monthly view):
    # Monthly metrics
    for col in ["indpro", "cfnai", "unrate", "cpi", "umcsent", "houst"]:
        mom, yoy = mom_yoy_pct_monthly(df[col])
        df[f"{col}_mom"] = mom
        df[f"{col}_yoy"] = yoy

    # Daily-derived monthly metrics
    for col in ["dgs10", "dgs3mo", "hy_oas", "vix", "sp500"]:
        mom, yoy = mom_yoy_pct_monthly(df[col])
        df[f"{col}_mom"] = mom
        df[f"{col}_yoy"] = yoy

    # Quarterly (in monthly view)
    g_mom, g_yoy = mom_yoy_pct_quarterly_month_view(df["gdp_saar"])
    df["gdp_saar_mom"] = g_mom
    df["gdp_saar_yoy"] = g_yoy

    # ---- Macro signals (Core 3)
    df["indpro_signal"] = (df["indpro_yoy"] > 0).astype(int)
    df["cfnai_signal"]  = (df["cfnai"] > 0).astype(int)
    df["yield_curve_signal"] = (df["yield_curve_spread"] > 0).astype(int)

    df["macro_score"] = (df["indpro_signal"] + df["cfnai_signal"] + df["yield_curve_signal"]).astype(int)
    df["macro_regime"] = np.where(df["macro_score"] >= 2, "expansion", "contraction")

    # ---- Risk signals (FRED)
    df["sp500_sma_10m"] = rolling_sma(df["sp500"], TREND_SMA_MONTHS)
    df["trend_on"] = (df["sp500"] >= df["sp500_sma_10m"]).astype(int)

    df["hy_oas_sma_12m"] = rolling_sma(df["hy_oas"], CREDIT_SMA_MONTHS)
    hy_barrier = pd.concat(
        [pd.Series(CREDIT_OAS_BPS_THRESHOLD, index=df.index), df["hy_oas_sma_12m"]],
        axis=1
    ).max(axis=1)
    df["credit_off"] = (df["hy_oas"] > hy_barrier).astype(int)

    df["vix_off"] = (df["vix"] > VIX_THRESHOLD).astype(int)
    df["off_votes"] = df["credit_off"] + df["vix_off"]

    def _risk_regime(row):
        if (row["trend_on"] == 1) and (row["off_votes"] == 0):
            return "On"
        if row["off_votes"] >= 2:
            return "Off"
        return "Mixed"

    df["risk_regime"] = df.apply(_risk_regime, axis=1)

    # ---- Order columns: date + regimes/scores first, then the rest
    df_out = df.copy().reset_index().rename(columns={"index": "date"})
    df_out["date"] = df_out["date"].dt.strftime("%Y-%m-%d")

    front_cols = [
        "date",
        "macro_regime", "macro_score", "indpro_signal", "cfnai_signal", "yield_curve_signal",
        "risk_regime", "off_votes", "trend_on", "credit_off", "vix_off",
    ]

    # Everything else, stable order
    rest_cols = [
        # Macro levels
        "indpro", "indpro_mom", "indpro_yoy",
        "cfnai", "cfnai_mom", "cfnai_yoy",
        "dgs10", "dgs10_mom", "dgs10_yoy",
        "dgs3mo", "dgs3mo_mom", "dgs3mo_yoy",
        "yield_curve_spread", "yield_curve_signal",
        "hy_oas", "hy_oas_mom", "hy_oas_yoy", "hy_oas_sma_12m",
        "unrate", "unrate_mom", "unrate_yoy",
        "cpi", "cpi_mom", "cpi_yoy",
        "gdp_saar", "gdp_saar_mom", "gdp_saar_yoy",
        "umcsent", "umcsent_mom", "umcsent_yoy",
        "houst", "houst_mom", "houst_yoy",
        "vix", "vix_mom", "vix_yoy",
        "sp500", "sp500_mom", "sp500_yoy", "sp500_sma_10m",
    ]

    # Keep only columns that exist (defensive in case of changes)
    rest_cols = [c for c in rest_cols if c in df_out.columns]
    df_out = df_out[front_cols + rest_cols]

    # ---- Interpolation report
    # Build masks aligned to idx for each metric (True where we interpolated/filled the monthly view)
    # Daily→Monthly masks (linear inside series)
    interp_masks = {
        "dgs10":  dgs10_interp.reindex(idx).fillna(False),
        "dgs3mo": dgs3m_interp.reindex(idx).fillna(False),
        "hy_oas": hy_interp.reindex(idx).fillna(False),
        "vix":    vix_interp.reindex(idx).fillna(False),
        "sp500":  spx_interp.reindex(idx).fillna(False),
        # Monthly masks
        "indpro": indpro_interp.reindex(idx).fillna(False),
        "cfnai":  cfnai_interp.reindex(idx).fillna(False),
        "unrate": unrate_interp.reindex(idx).fillna(False),
        "cpi":    cpi_interp.reindex(idx).fillna(False),
        "umcsent": umcs_interp.reindex(idx).fillna(False),
        "houst":   houst_interp.reindex(idx).fillna(False),
        # Quarterly mask (months not at quarter-ends are "filled")
        "gdp_saar": gdp_interp.reindex(idx).fillna(True),
    }

    total_months = len(idx)
    rows = []
    for k, mask in interp_masks.items():
        filled = int(mask.sum())
        pct = (filled / total_months) * 100.0
        rows.append({"metric": k, "filled_count": filled, "filled_pct": pct, "total_months": total_months})
    interp_report = pd.DataFrame(rows).sort_values("metric")
    interp_report.to_csv(INTERP_REPORT_PATH, index=False)

    # ---- Write main CSV (overwrite full file)
    df_out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(df_out)} rows (latest {df_out['date'].iloc[-1]}).")
    print(f"Wrote interpolation audit to {INTERP_REPORT_PATH}")

if __name__ == "__main__":
    main()