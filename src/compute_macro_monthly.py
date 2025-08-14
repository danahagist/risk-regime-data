# src/compute_macro_monthly.py
# Macro + base risk series (FRED-only), monthly pipeline
# - Fetch from FRED
# - Daily/Monthly/Quarterly -> Monthly (month-end sample -> stamp to 1st of month)
# - Forward-fill so each month has a value
# - Compute MoM% and YoY% on the filled levels
# - Core 3 Macro signals: INDPRO YoY>0, LEI YoY>0, YieldCurve>0 (bps)
# - Output: data/macro_monthly_history.csv (append + de-dup by date)

import os
import sys
import time
import pandas as pd
from fredapi import Fred

# -------------------------------
# Config
# -------------------------------
FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    print("ERROR: FRED_API_KEY is not set in the environment.", file=sys.stderr)
    sys.exit(1)

fred = Fred(api_key=FRED_API_KEY)

OUT_PATH   = "data/macro_monthly_history.csv"
START_DATE = "1990-01-01"

# FRED series codes
INDPRO   = "INDPRO"            # Industrial Production: Total Index (level)
LEI      = "USSLIND"           # Leading Index for the United States (level)
DGS10    = "DGS10"             # 10Y Treasury Constant Maturity (percent)
DGS3MO   = "DGS3MO"            # 3M Treasury Constant Maturity (percent)
HY_OAS   = "BAMLH0A0HYM2"      # ICE BofA US High Yield OAS (bps)
UNRATE   = "UNRATE"            # Unemployment Rate (%)
CPI      = "CPIAUCSL"          # CPI All Urban Consumers (Index, SA)
GDP_SAAR = "A191RL1Q225SBEA"   # Real GDP, QoQ % SAAR (quarterly rate, %)
UMCSENT  = "UMCSENT"           # University of Michigan Consumer Sentiment (level)
HOUST    = "HOUST"             # Housing Starts (SAAR, thousands)
SP500    = "SP500"             # S&P 500 index level

# -------------------------------
# Helpers
# -------------------------------
def safe_get_series(code: str, start: str) -> pd.Series:
    """Fetch a FRED series with basic retries. Returns Series with DatetimeIndex."""
    last_err = None
    for i in range(4):
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

def to_monthly_ffill(series: pd.Series) -> pd.Series:
    """
    Convert daily/monthly/quarterly to monthly:
      - resample to month-end and take the last observation,
      - convert to Period(M) then to Timestamp at month start (YYYY-MM-01),
      - forward-fill so every month has a value.
    """
    if series is None or series.empty:
        return pd.Series(dtype="float64")
    s = pd.Series(series).dropna().sort_index()
    # month-end sample (M) -> take last obs each month
    s = s.resample("M").last()
    # stamp to the 1st of month (start)
    s.index = s.index.to_period("M").to_timestamp(how="start")
    # forward-fill to remove holes (e.g., quarterly GDP between quarters)
    return s.ffill()

def pct_mom_yoy_ffilled(level: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Percent MoM and YoY on a forward-filled level series (in %)."""
    l = level.astype(float)
    mom = l.pct_change(1) * 100.0
    yoy = l.pct_change(12) * 100.0
    return mom, yoy

# -------------------------------
# Main
# -------------------------------
def main():
    # 1) Fetch raw series from FRED
    indpro_raw   = safe_get_series(INDPRO,   START_DATE)   # level
    lei_raw      = safe_get_series(LEI,      START_DATE)   # level
    dgs10_raw    = safe_get_series(DGS10,    START_DATE)   # percent
    dgs3mo_raw   = safe_get_series(DGS3MO,   START_DATE)   # percent
    hy_oas_raw   = safe_get_series(HY_OAS,   START_DATE)   # bps
    unrate_raw   = safe_get_series(UNRATE,   START_DATE)   # %
    cpi_raw      = safe_get_series(CPI,      START_DATE)   # index
    gdp_raw      = safe_get_series(GDP_SAAR, START_DATE)   # % q/q SAAR (quarterly)
    umcsent_raw  = safe_get_series(UMCSENT,  START_DATE)   # level
    houst_raw    = safe_get_series(HOUST,    START_DATE)   # SAAR (thousands)
    sp500_raw    = safe_get_series(SP500,    START_DATE)   # level (daily)

    # 2) Convert all to monthly (first-of-month) and forward-fill
    indpro_m   = to_monthly_ffill(indpro_raw).rename("indpro")
    lei_m      = to_monthly_ffill(lei_raw).rename("lei")
    dgs10_m    = to_monthly_ffill(dgs10_raw).rename("dgs10")
    dgs3mo_m   = to_monthly_ffill(dgs3mo_raw).rename("dgs3mo")
    hy_oas_m   = to_monthly_ffill(hy_oas_raw).rename("hy_oas")
    unrate_m   = to_monthly_ffill(unrate_raw).rename("unrate")
    cpi_m      = to_monthly_ffill(cpi_raw).rename("cpi")
    gdp_m      = to_monthly_ffill(gdp_raw).rename("gdp_saar")
    umcsent_m  = to_monthly_ffill(umcsent_raw).rename("umcsent")
    houst_m    = to_monthly_ffill(houst_raw).rename("houst")
    sp500_m    = to_monthly_ffill(sp500_raw).rename("sp500")

    # 3) Outer join to a single monthly frame, then ffill once more (alignment polish)
    df = pd.concat(
        [indpro_m, lei_m, dgs10_m, dgs3mo_m, hy_oas_m, unrate_m, cpi_m,
         gdp_m, umcsent_m, houst_m, sp500_m],
        axis=1
    ).sort_index().ffill()

    # 4) Derived fields
    # Yield curve spread in basis points
    df["yield_curve_spread"] = (df["dgs10"] - df["dgs3mo"]) * 100.0

    # Percent changes (MoM/YoY) on levels
    df["indpro_mom"],   df["indpro_yoy"]   = pct_mom_yoy_ffilled(df["indpro"])
    df["lei_mom"],      df["lei_yoy"]      = pct_mom_yoy_ffilled(df["lei"])
    df["yc_mom"],       df["yc_yoy"]       = pct_mom_yoy_ffilled(df["yield_curve_spread"])
    df["hy_oas_mom"],   df["hy_oas_yoy"]   = pct_mom_yoy_ffilled(df["hy_oas"])
    df["unrate_mom"],   df["unrate_yoy"]   = pct_mom_yoy_ffilled(df["unrate"])
    df["cpi_mom"],      df["cpi_yoy"]      = pct_mom_yoy_ffilled(df["cpi"])
    df["gdp_mom"],      df["gdp_yoy"]      = pct_mom_yoy_ffilled(df["gdp_saar"])
    df["umcsent_mom"],  df["umcsent_yoy"]  = pct_mom_yoy_ffilled(df["umcsent"])
    df["houst_mom"],    df["houst_yoy"]    = pct_mom_yoy_ffilled(df["houst"])
    df["sp500_mom"],    df["sp500_yoy"]    = pct_mom_yoy_ffilled(df["sp500"])

    # 5) Core 3 Macro signals
    df["indpro_signal"]      = (df["indpro_yoy"] > 0).astype(float)
    df["lei_signal"]         = (df["lei_yoy"] > 0).astype(float)
    df["yield_curve_signal"] = (df["yield_curve_spread"] > 0).astype(float)

    df["macro_score"]  = df[["indpro_signal", "lei_signal", "yield_curve_signal"]].sum(axis=1)
    df["macro_regime"] = df["macro_score"].apply(lambda x: "expansion" if x >= 2 else "contraction")

    # 6) Final formatting
    out = df.copy()
    out = out.reset_index().rename(columns={"index": "date"})
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")  # always YYYY-MM-01

    # Column order: levels, changes, signals, scores
    cols = [
        "date",
        "indpro", "indpro_mom", "indpro_yoy", "indpro_signal",
        "lei", "lei_mom", "lei_yoy", "lei_signal",
        "dgs10", "dgs3mo", "yield_curve_spread", "yc_mom", "yc_yoy", "yield_curve_signal",
        "hy_oas", "hy_oas_mom", "hy_oas_yoy",
        "unrate", "unrate_mom", "unrate_yoy",
        "cpi", "cpi_mom", "cpi_yoy",
        "gdp_saar", "gdp_mom", "gdp_yoy",
        "umcsent", "umcsent_mom", "umcsent_yoy",
        "houst", "houst_mom", "houst_yoy",
        "sp500", "sp500_mom", "sp500_yoy",
        "macro_score", "macro_regime",
    ]
    out = out[cols]

    # 7) Append + de-duplicate by date
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    if os.path.exists(OUT_PATH):
        try:
            prev = pd.read_csv(OUT_PATH, dtype=str)
            combined = pd.concat([prev, out.astype(str)], ignore_index=True)
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