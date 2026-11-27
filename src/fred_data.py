"""FRED series catalog and fetch helpers.

Each series carries a `rights` flag:
  public       produced by a US government agency or Federal Reserve bank
  third_party  copyrighted by another owner and relayed by FRED. Fine for
               personal use; needs the owner's permission before a public or
               commercial launch (see README, "Data rights").
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

SERIES = {
    # growth and labor
    "CFNAI": ("Chicago Fed National Activity Index", "public"),
    "INDPRO": ("Industrial production", "public"),
    "UNRATE": ("Unemployment rate", "public"),
    "SAHMREALTIME": ("Sahm rule (real time)", "public"),
    "ICSA": ("Initial jobless claims", "public"),
    "NROU": ("Natural rate of unemployment (CBO)", "public"),
    # inflation
    "PCEPILFE": ("Core PCE price index", "public"),
    "CPIAUCSL": ("CPI, all items", "public"),
    # policy and rates
    "DFEDTARU": ("Fed funds target, upper bound", "public"),
    "FEDFUNDS": ("Effective fed funds rate", "public"),
    "DGS10": ("10-year Treasury yield", "public"),
    "DGS2": ("2-year Treasury yield", "public"),
    "DGS3MO": ("3-month Treasury yield", "public"),
    "NFCI": ("Chicago Fed financial conditions index", "public"),
    "NFCICREDIT": ("Chicago Fed financial conditions, credit subindex", "public"),
    # valuation
    "NCBEILQ027S": ("Corporate equities, market value (Fed Z.1)", "public"),
    "GDP": ("Nominal GDP", "public"),
    "DFII10": ("10-year real Treasury yield (TIPS)", "public"),
    # liquidity
    "WALCL": ("Federal Reserve total assets", "public"),
    "WTREGEN": ("Treasury General Account", "public"),
    "RRPONTSYD": ("Overnight reverse repo balances", "public"),
    # markets (third-party owners)
    "SP500": ("S&P 500 index", "third_party"),
    "VIXCLS": ("CBOE VIX", "third_party"),
    "VXVCLS": ("CBOE 3-month VIX", "third_party"),
    "BAMLH0A0HYM2": ("ICE BofA US high-yield spread", "third_party"),
}

START_DATE = "1990-01-01"


def fetch_all(start: str = START_DATE) -> dict[str, pd.Series]:
    """Fetch every catalog series from FRED. A series that fails comes back empty,
    so one outage does not sink the whole run; votes that need it show as no data."""
    from fredapi import Fred

    key = os.getenv("FRED_API_KEY")
    if not key:
        print("ERROR: FRED_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)
    fred = Fred(api_key=key)

    out: dict[str, pd.Series] = {}
    for code in SERIES:
        out[code] = _fetch_one(fred, code, start)
    return out


def _fetch_one(fred, code: str, start: str) -> pd.Series:
    last_err = None
    for i in range(4):
        try:
            s = pd.Series(fred.get_series(code, observation_start=start), dtype="float64")
            s.index = pd.to_datetime(s.index)
            return s.dropna().sort_index()
        except Exception as e:  # network or API error: retry, then give up on this series
            last_err = e
            time.sleep(2 * i)
    print(f"[fred] giving up on {code}: {last_err}", file=sys.stderr)
    return pd.Series(dtype="float64")


def load_fixture(directory: str) -> dict[str, pd.Series]:
    """Load series from <directory>/<KEY>.csv (columns: date,value). Used by tests and
    offline previews. A key such as PX:SPY is stored as PX_SPY.csv."""
    out: dict[str, pd.Series] = {code: pd.Series(dtype="float64") for code in SERIES}
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".csv"):
            continue
        key = name[:-4]
        for prefix in ("PX", "SHILLER", "AAII"):
            if key.startswith(prefix + "_"):
                key = prefix + ":" + key[len(prefix) + 1:]
        df = pd.read_csv(os.path.join(directory, name), parse_dates=["date"])
        out[key] = pd.Series(df["value"].values, index=df["date"], dtype="float64").dropna().sort_index()
    return out
