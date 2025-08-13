# src/test_fred.py
import os
from datetime import date
from fredapi import Fred

START = "1990-01-01"

def main():
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY not set in environment")

    fred = Fred(api_key=api_key)

    # Core series we actually pull now
    series_ids = [
        "INDPRO",   # Industrial Production (replacement for PMI)
        "USSLIND",  # Leading Index
        "DGS10",    # 10Y Treasury
        "DGS3MO",   # 3M Treasury
    ]

    for sid in series_ids:
        s = fred.get_series(sid, observation_start=START)
        print(f"{sid} rows: {len(s)}")
        if len(s) == 0:
            raise AssertionError(f"{sid} returned no data")

    # Optional: quick sanity check that we can compute the spread
    dgs10 = fred.get_series("DGS10", observation_start=START)
    dgs3m = fred.get_series("DGS3MO", observation_start=START)
    spread = dgs10 - dgs3m
    print(f"Yield curve spread rows: {len(spread)}; last={spread.dropna().iloc[-1]:.2f}")

if __name__ == "__main__":
    main()
