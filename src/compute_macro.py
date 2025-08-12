# src/compute_macro.py
import os, time
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

OUT_PATH = "macro_history.csv"
START = "1980-01-01"

FRED = {
    # Regime trio
    "PMI":      "NAPM",
    "LEI":      "USSLIND",
    "YC_10Y3M": "T10Y3M",
    # Health dashboard
    "HY_OAS":   "BAMLH0A0HYM2",
    "BAA10Y":   "BAA10Y",
    "UNRATE":   "UNRATE",
    "ICSA":     "ICSA",
    "CPI":      "CPIAUCSL",
    "UMCSENT":  "UMCSENT",
    "RSAFS":    "RSAFS",
}

API_KEY = os.getenv("FRED_API_KEY", "").strip()

def fred_series(code, start=START):
    """
    When API key exists, use the official FRED API (DataReader with api_key).
    Do NOT fall back to fredgraph.csv if a key is present.
    Without a key, use the legacy CSV endpoint with retries.
    """
    last_err = None

    # 1) Preferred: official API
    if API_KEY:
        for i in range(3):
            try:
                # pandas-datareader will hit api.stlouisfed.org when api_key is provided
                df = pdr.DataReader(code, "fred", start=start, api_key=API_KEY)
                s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
                s = s.dropna()
                s.name = code
                print(f"[FRED API] {code}: {len(s)} rows")
                return s
            except Exception as e:
                last_err = e
                print(f"[FRED API] attempt {i+1} failed for {code}: {e}")
                time.sleep(2*(i+1))
        # If we get here, API failed 3x -> raise (don’t hit CSV when key exists)
        raise last_err

    # 2) No key: legacy CSV path (fredgraph.csv)
    for i in range(3):
        try:
            df = pdr.DataReader(code, "fred", start=start)
            s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
            s = s.dropna()
            s.name = code
            print(f"[FRED CSV] {code}: {len(s)} rows")
            return s
        except Exception as e:
            last_err = e
            print(f"[FRED CSV] attempt {i+1} failed for {code}: {e}")
            time.sleep(2*(i+1))
    raise last_err

def month_end_align(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.index.freq is None:
        s = s.resample("M").last()
    return s

def weekly_to_monthly_last(s: pd.Series) -> pd.Series:
    s = s.asfreq("W-FRI")
    return s.resample("M").last()

def pct_yoy(s: pd.Series) -> pd.Series:
    return s.pct_change(12) * 100.0

def zscore_full(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)

def main():
    # 1) Pull series
    raw = {}
    for label, code in FRED.items():
        s = fred_series(code)
        s = weekly_to_monthly_last(s) if label == "ICSA" else month_end_align(s)
        raw[label] = s

    # 2) Combine monthly panel; ffill publication lags
    m = pd.concat(raw, axis=1).sort_index().ffill()

    # 3) Derived measures
    lei_yoy   = pct_yoy(m["LEI"])
    cpi_yoy   = pct_yoy(m["CPI"])
    rsafs_yoy = pct_yoy(m["RSAFS"])

    pmi_score = (m["PMI"] > 50).astype(int).where(True, -1)
    lei_score = (lei_yoy > 0).astype(int).where(True, -1)
    yc_score  = (m["YC_10Y3M"] > 0).astype(int).where(True, -1)
    regime_total = pmi_score + lei_score + yc_score
    macro_regime = np.where(regime_total >= 2, "Expansion",
                     np.where(regime_total <= -2, "Contraction", "Borderline"))

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

    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH} with {len(out)} monthly rows.")

if __name__ == "__main__":
    main()