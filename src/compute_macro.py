# src/compute_macro.py
import os, time
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

OUT_PATH = "macro_history.csv"
START = "1980-01-01"

# FRED series (codes)
FRED = {
    # Regime trio
    "PMI":      "NAPM",        # ISM Manufacturing PMI (index, monthly)
    "LEI":      "USSLIND",     # Leading Economic Index (index, monthly)
    "YC_10Y3M": "T10Y3M",      # 10Y - 3M spread (%)

    # Health dashboard
    "HY_OAS":   "BAMLH0A0HYM2",# HY OAS (%)
    "BAA10Y":   "BAA10Y",      # Moody's Baa - 10Y Treasury spread (%)
    "UNRATE":   "UNRATE",      # Unemployment rate (%)
    "ICSA":     "ICSA",        # Initial jobless claims (weekly)
    "CPI":      "CPIAUCSL",    # CPI (index, monthly)
    "UMCSENT":  "UMCSENT",     # U. Michigan Consumer Sentiment (index)
    "RSAFS":    "RSAFS",       # Retail Sales, Advance (level, monthly)
}

def fred_series(code, start=START):
    """Download a single FRED series via pandas_datareader with simple retries."""
    last_err = None
    for i in range(3):
        try:
            s = pdr.DataReader(code, "fred", start=start)
            s = s.iloc[:,0] if isinstance(s, pd.DataFrame) else s
            s = s.dropna()
            s.name = code
            return s
        except Exception as e:
            last_err = e
            time.sleep(2*(i+1))
    raise last_err

def month_end_align(s: pd.Series) -> pd.Series:
    """Ensure monthly month-end index."""
    s = s.dropna()
    if s.index.freq is None:
        s = s.resample("M").last()
    return s

def weekly_to_monthly_last(s: pd.Series) -> pd.Series:
    """Weekly series → month-end by taking the last weekly print each month."""
    s = s.asfreq("W-FRI")
    s = s.resample("M").last()
    return s

def pct_yoy(s: pd.Series) -> pd.Series:
    return s.pct_change(12) * 100.0

def zscore_full(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

def main():
    # 1) Pull all series
    raw = {}
    for label, code in FRED.items():
        s = fred_series(code)
        if label == "ICSA":  # weekly → monthly last
            s = weekly_to_monthly_last(s)
        else:
            s = month_end_align(s)
        raw[label] = s

    # 2) Combine monthly panel; ffill for publication lag
    m = pd.concat(raw, axis=1).sort_index().ffill()

    # 3) Derived measures
    lei_yoy   = pct_yoy(m["LEI"])
    cpi_yoy   = pct_yoy(m["CPI"])
    rsafs_yoy = pct_yoy(m["RSAFS"])

    # Regime scoring (simple, transparent)
    pmi_score = np.where(m["PMI"] > 50, 1, -1)
    lei_score = np.where(lei_yoy > 0, 1, -1)
    yc_score  = np.where(m["YC_10Y3M"] > 0, 1, -1)
    regime_total = pmi_score + lei_score + yc_score
    macro_regime = np.where(regime_total >= 2, "Expansion",
                     np.where(regime_total <= -2, "Contraction", "Borderline"))

    # Health composite (equal-weight z-scores; sign so higher=better)
    health_inputs = pd.DataFrame({
        "PMI_z":        zscore_full(m["PMI"]),
        "LEI_yoy_z":    zscore_full(lei_yoy),
        "YC_10Y3M_z":   zscore_full(m["YC_10Y3M"]),
        "HY_OAS_z":    -zscore_full(m["HY_OAS"]),
        "BAA10Y_z":    -zscore_full(m["BAA10Y"]),
        "UNRATE_z":    -zscore_full(m["UNRATE"]),
        "ICSA_z":      -zscore_full(m["ICSA"]),
        "CPI_yoy_z":   -zscore_full(cpi_yoy),
        "UMCSENT_z":    zscore_full(m["UMCSENT"]),
        "RSAFS_yoy_z":  zscore_full(rsafs_yoy),
    }, index=m.index).dropna(how="all")
    health_composite = health_inputs.mean(axis=1)

    # 4) Assemble output table
    out = pd.DataFrame(index=m.index)
    out["pmi"]        = m["PMI"]
    out["lei"]        = m["LEI"]
    out["yc_10y3m"]   = m["YC_10Y3M"]
    out["lei_yoy"]    = lei_yoy
    out["unrate"]     = m["UNRATE"]
    out["icsa"]       = m["ICSA"]
    out["cpi_yoy"]    = cpi_yoy
    out["umcsent"]    = m["UMCSENT"]
    out["rsafs_yoy"]  = rsafs_yoy
    out["hy_oas"]     = m["HY_OAS"]
    out["baa10y"]     = m["BAA10Y"]

    out["pmi_score"]  = pmi_score
    out["lei_score"]  = lei_score
    out["yc_score"]   = yc_score
    out["macro_regime_score"] = regime_total
    out["macro_regime"] = macro_regime
    out["health_composite"] = health_composite

    out = out.dropna(subset=["pmi","lei","yc_10y3m"])
    out = out.reset_index().rename(columns={"index":"date"})
    out["date"] = pd.to_datetime(out["date"]).dt.date.astype(str)

    # 5) Write full history (overwrite each run)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(out)} monthly rows.")

if __name__ == "__main__":
    main()