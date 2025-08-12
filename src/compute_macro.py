# src/compute_macro.py
import os
import time
from datetime import datetime, date
import pandas as pd
from fredapi import Fred

FRED_API_KEY = os.getenv("FRED_API_KEY")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY not set in environment.")

fred = Fred(api_key=FRED_API_KEY)

# Output file (weekly history)
OUT_PATH = "data/macro_weekly_history.csv"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

# Start date for long history
START = "1990-01-01"

# Live FRED series:
# PMI: S&P Global US Manufacturing PMI
PMI_CODE = "USPMI"
# LEI: Philly Fed Leading Index for the United States
LEI_CODE = "USSLIND"
# Yield curve components: 10Y constant maturity minus 3M T-Bill
DGS10_CODE = "DGS10"
DGS3MO_CODE = "DGS3MO"

def get_series(code: str, start: str) -> pd.Series:
    """Fetch a single FRED series with basic retry."""
    last_err = None
    for i, wait in enumerate([0, 2, 4], start=1):
        try:
            s = fred.get_series(code, observation_start=start)
            if s is None or len(s) == 0:
                raise ValueError(f"{code} returned empty series")
            s = s.dropna()
            s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            return s
        except Exception as e:
            last_err = e
            print(f"[fredapi] attempt {i} failed for {code}: {e} (sleep {wait}s)")
            time.sleep(wait)
    raise last_err

def weeklyize_last(s: pd.Series) -> pd.Series:
    """
    Convert any frequency series to WEEKLY (Friday) using last observation available.
    Forward-fill so the latest monthly/daily value carries through to each week.
    """
    s = s.asfreq("D")  # daily grid
    s = s.ffill()
    # Use Friday as weekly anchor to match market-style weeks
    s_w = s.resample("W-FRI").last()
    return s_w

def classify_macro(pmi: float, lei: float, yc: float) -> str:
    """
    Expansion/Contraction by simple majority of favorable signals:
    - PMI favorable if >= 50
    - LEI favorable if > 0 (Philly Fed leading index is MoM annualized; >0 = positive momentum)
    - Yield curve favorable if spread > 0
    """
    votes = 0
    votes += 1 if pmi is not None and pmi >= 50 else 0
    votes += 1 if lei is not None and lei > 0 else 0
    votes += 1 if yc is not None and yc > 0 else 0
    return "Expansion" if votes >= 2 else "Contraction"

def main():
    # Fetch base series
    pmi = get_series(PMI_CODE, START)        # level
    lei = get_series(LEI_CODE, START)        # level (Philly Fed leading index)
    dgs10 = get_series(DGS10_CODE, START)    # percent
    dgs3m = get_series(DGS3MO_CODE, START)   # percent

    # Compute yield curve (10y - 3m)
    yc = (dgs10 - dgs3m).dropna()
    # Align to weekly
    pmi_w = weeklyize_last(pmi.rename("pmi"))
    lei_w = weeklyize_last(lei.rename("lei"))
    yc_w = weeklyize_last(yc.rename("yield_curve"))

    # Combine
    df = pd.concat([pmi_w, lei_w, yc_w], axis=1).dropna(how="all")
    # Simple per-row classification
    df["macro_environment"] = df.apply(
        lambda r: classify_macro(r.get("pmi"), r.get("lei"), r.get("yield_curve")),
        axis=1
    )

    # Keep only the last week to append to history (same pattern as your risk file)
    last_row = df.iloc[[-1]].copy()
    out_row = pd.DataFrame([{
        "date": last_row.index[-1].date().isoformat(),
        "pmi": float(last_row["pmi"].iloc[0]) if pd.notna(last_row["pmi"].iloc[0]) else None,
        "lei": float(last_row["lei"].iloc[0]) if pd.notna(last_row["lei"].iloc[0]) else None,
        "yield_curve": float(last_row["yield_curve"].iloc[0]) if pd.notna(last_row["yield_curve"].iloc[0]) else None,
        "macro_environment": last_row["macro_environment"].iloc[0]
    }])

    # Initialize file with header if missing
    if not os.path.exists(OUT_PATH):
        out_row.to_csv(OUT_PATH, index=False)
    else:
        # Append if this date isn’t already present
        hist = pd.read_csv(OUT_PATH)
        if "date" not in hist.columns:
            hist = pd.DataFrame(columns=["date", "pmi", "lei", "yield_curve", "macro_environment"])
        if not (hist["date"] == out_row["date"].iloc[0]).any():
            out_row.to_csv(OUT_PATH, mode="a", header=False, index=False)
        else:
            # Replace existing last row for idempotency (optional)
            hist = hist[hist["date"] != out_row["date"].iloc[0]]
            hist = pd.concat([hist, out_row], ignore_index=True).sort_values("date")
            hist.to_csv(OUT_PATH, index=False)

    # Also print the latest values to logs for quick visibility
    print("Latest macro:")
    print(out_row.to_string(index=False))

if __name__ == "__main__":
    main()