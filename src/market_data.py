"""Data that FRED does not carry: ETF prices, long-run valuation series and a sentiment survey.

Everything here belongs to third parties (see README, "Data rights"). Every fetch
is best effort: if a source is down or blocks the request, the series comes back
empty and the frameworks that need it show "No data" for the month.

Keys in the returned dict:
    PX:<TICKER>     daily adjusted close
    SHILLER:CAPE    monthly cyclically adjusted P/E
    SHILLER:PE      monthly trailing P/E
    AAII:SPREAD     weekly bulls minus bears, in percentage points
"""
from __future__ import annotations

import io
import re
import sys

import pandas as pd

SECTORS = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer discretionary", "XLF": "Financials",
    "XLI": "Industrials", "XLB": "Materials", "XLE": "Energy", "XLV": "Health care",
    "XLP": "Consumer staples", "XLU": "Utilities", "XLRE": "Real estate",
}
CYCLICAL = {"XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE"}
ASSET_ETFS = {
    "SPY": "US stocks", "ACWX": "Foreign stocks", "AGG": "US bonds", "GLD": "Gold", "DBC": "Commodities",
}
OTHER = {
    "BIL": "T-bills", "IWM": "US small caps", "RSP": "S&P 500 equal weight",
    "DIA": "Dow industrials", "IYT": "Dow transports", "CPER": "Copper",
}
TICKERS = {**ASSET_ETFS, **OTHER, **SECTORS}

SOURCES = [
    {"id": "ETF prices", "name": f"Adjusted closes for {len(TICKERS)} ETFs (Yahoo Finance, Stooq as fallback)", "rights": "third_party", "prefix": "PX:"},
    {"id": "Shiller CAPE and P/E", "name": "S&P 500 valuation history (multpl.com, Yale as fallback)", "rights": "third_party", "prefix": "SHILLER:"},
    {"id": "AAII survey", "name": "AAII investor sentiment survey", "rights": "third_party", "prefix": "AAII:"},
]

UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
START = "2003-01-01"
EMPTY = pd.Series(dtype="float64")


def _log(msg: str) -> None:
    print(f"[market_data] {msg}", file=sys.stderr)


def _get(url: str, timeout: int = 30):
    import requests
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    return r


# ------------------------------------------------------------------ prices
def _prices_yahoo(tickers: list[str]) -> dict[str, pd.Series]:
    import yfinance as yf
    df = yf.download(tickers, start=START, auto_adjust=True, progress=False, threads=False)
    if df is None or df.empty:
        return {}
    close = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df[["Close"]].rename(columns={"Close": tickers[0]})
    out = {}
    for t in close.columns:
        s = pd.to_numeric(close[t], errors="coerce").dropna()
        if len(s) > 250:
            s.index = pd.to_datetime(s.index).tz_localize(None)
            out[str(t)] = s.sort_index()
    return out


def _price_stooq(ticker: str) -> pd.Series:
    text = _get(f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d").text
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError("unexpected response")
    s = pd.Series(pd.to_numeric(df["Close"], errors="coerce").values, index=pd.to_datetime(df["Date"])).dropna()
    return s.loc[START:].sort_index()


def fetch_prices(tickers: list[str] | None = None) -> dict[str, pd.Series]:
    tickers = list(TICKERS) if tickers is None else tickers
    got: dict[str, pd.Series] = {}
    try:
        got = _prices_yahoo(tickers)
        _log(f"Yahoo returned {len(got)} of {len(tickers)} tickers")
    except Exception as e:
        _log(f"Yahoo failed: {type(e).__name__}: {e}")
    missing = [t for t in tickers if t not in got]
    stooq_failures = 0
    for t in missing:
        if stooq_failures >= 3:      # the source is down or blocking us; stop asking
            break
        try:
            s = _price_stooq(t)
            if len(s) > 250:
                got[t] = s
        except Exception as e:
            stooq_failures += 1
            _log(f"Stooq failed for {t}: {type(e).__name__}: {e}")
    if missing:
        _log(f"after Stooq fallback: {len(got)} of {len(tickers)} tickers")
    return {f"PX:{t}": got.get(t, EMPTY) for t in tickers}


# ------------------------------------------------------------------ valuation
def _num(x) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group()) if m else None


def _multpl(path: str) -> pd.Series:
    html = _get(f"https://www.multpl.com/{path}/table/by-month").text
    df = pd.read_html(io.StringIO(html))[0]
    df.columns = [str(c).strip() for c in df.columns]
    dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    vals = df.iloc[:, 1].map(_num)
    s = pd.Series(vals.values, index=dates, dtype="float64").dropna()
    s = s[s.index.notna()].sort_index()
    return s.groupby(s.index.to_period("M")).last().to_timestamp("M")


def _yale() -> tuple[pd.Series, pd.Series]:
    raw = _get("http://www.econ.yale.edu/~shiller/data/ie_data.xls", timeout=60).content
    df = pd.read_excel(io.BytesIO(raw), sheet_name="Data", header=7)
    df.columns = [str(c).strip() for c in df.columns]
    d = pd.to_numeric(df["Date"], errors="coerce")
    ok = d.notna()
    year = d[ok].astype(int)
    month = ((d[ok] - year) * 100).round().astype(int).clip(1, 12)
    idx = pd.to_datetime(dict(year=year, month=month, day=1)) + pd.offsets.MonthEnd(0)
    cape_col = next(c for c in df.columns if c.upper().startswith("CAPE"))
    cape = pd.Series(pd.to_numeric(df.loc[ok, cape_col], errors="coerce").values, index=idx).dropna()
    pe = pd.Series((pd.to_numeric(df.loc[ok, "P"], errors="coerce") / pd.to_numeric(df.loc[ok, "E"], errors="coerce")).values, index=idx).dropna()
    return cape, pe


def fetch_shiller() -> dict[str, pd.Series]:
    out = {"SHILLER:CAPE": EMPTY, "SHILLER:PE": EMPTY}
    for key, path in (("SHILLER:CAPE", "shiller-pe"), ("SHILLER:PE", "s-p-500-pe-ratio")):
        try:
            s = _multpl(path)
            if len(s) > 240:
                out[key] = s
        except Exception as e:
            _log(f"multpl {path} failed: {type(e).__name__}: {e}")
    if out["SHILLER:CAPE"].empty or out["SHILLER:PE"].empty:
        try:
            cape, pe = _yale()
            if out["SHILLER:CAPE"].empty:
                out["SHILLER:CAPE"] = cape
            if out["SHILLER:PE"].empty:
                out["SHILLER:PE"] = pe
        except Exception as e:
            _log(f"Yale file failed: {type(e).__name__}: {e}")
    _log(f"valuation: CAPE {len(out['SHILLER:CAPE'])} months, P/E {len(out['SHILLER:PE'])} months")
    return out


# ------------------------------------------------------------------ sentiment
def fetch_aaii() -> dict[str, pd.Series]:
    try:
        raw = _get("https://www.aaii.com/files/surveys/sentiment.xls", timeout=60).content
        df = pd.read_excel(io.BytesIO(raw), sheet_name=0, header=None)
        # find the header row: the one that mentions both Bullish and Bearish
        hdr = next(i for i in range(min(10, len(df))) if {"bullish", "bearish"} <= {str(x).strip().lower() for x in df.iloc[i]})
        cols = [str(x).strip().lower() for x in df.iloc[hdr]]
        body = df.iloc[hdr + 1:]
        dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
        bull = pd.to_numeric(body.iloc[:, cols.index("bullish")], errors="coerce")
        bear = pd.to_numeric(body.iloc[:, cols.index("bearish")], errors="coerce")
        spread = (bull - bear) * (100.0 if bull.dropna().abs().median() <= 1.0 else 1.0)
        s = pd.Series(spread.values, index=dates).dropna()
        s = s[s.index.notna()].sort_index()
        _log(f"AAII: {len(s)} weeks, latest {s.index[-1].date() if len(s) else 'none'}")
        return {"AAII:SPREAD": s}
    except Exception as e:
        _log(f"AAII failed: {type(e).__name__}: {e}")
        return {"AAII:SPREAD": EMPTY}


def fetch_all() -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for fn in (fetch_prices, fetch_shiller, fetch_aaii):
        try:
            out.update(fn())
        except Exception as e:  # never let an outside source stop the monthly run
            _log(f"{fn.__name__} failed: {type(e).__name__}: {e}")
    return out
