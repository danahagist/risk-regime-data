"""Parsing tests for the outside data sources, with the network replaced by canned responses."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import fred_data  # noqa: E402
import market_data as md  # noqa: E402


class Resp:
    def __init__(self, text="", content=b""):
        self.text, self.content = text, content


def multpl_html(rows):
    body = "".join(f"<tr><td>{d}</td><td>{v}</td></tr>" for d, v in rows)
    return f"<html><table id='datatable'><tr><th>Date</th><th>Value</th></tr>{body}</table></html>"


def test_multpl_table_parses_dates_values_and_estimate_marks(monkeypatch):
    rows = [("Nov 27, 2026", "†\n38.12"), ("Nov 1, 2026", "39.37"), ("Oct 1, 2026", "40.50"), ("Sep 1, 2026", "39.9 estimate")]
    monkeypatch.setattr(md, "_get", lambda url, timeout=30: Resp(text=multpl_html(rows)))
    s = md._multpl("shiller-pe")
    assert list(s.index) == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-31"), pd.Timestamp("2026-11-30")]
    assert s.iloc[-1] == pytest.approx(38.12)      # the latest reading within the month wins
    assert s.iloc[0] == pytest.approx(39.9)


def test_stooq_csv_parses(monkeypatch):
    dates = pd.bdate_range("2024-01-01", periods=400)
    csv = "Date,Open,High,Low,Close,Volume\n" + "\n".join(f"{d.date()},1,1,1,{100 + i},10" for i, d in enumerate(dates))
    monkeypatch.setattr(md, "_get", lambda url, timeout=30: Resp(text=csv))
    s = md._price_stooq("SPY")
    assert len(s) == 400 and s.iloc[-1] == 499


def test_stooq_rejects_an_error_page(monkeypatch):
    monkeypatch.setattr(md, "_get", lambda url, timeout=30: Resp(text="No data"))
    with pytest.raises(Exception):
        md._price_stooq("SPY")


def test_prices_fall_back_and_never_raise(monkeypatch):
    def boom(tickers):
        raise RuntimeError("rate limited")
    dates = pd.bdate_range("2024-01-01", periods=300)
    csv = "Date,Open,High,Low,Close,Volume\n" + "\n".join(f"{d.date()},1,1,1,50,10" for d in dates)
    monkeypatch.setattr(md, "_prices_yahoo", boom)
    monkeypatch.setattr(md, "_get", lambda url, timeout=30: Resp(text=csv))
    out = md.fetch_prices(["SPY", "AGG"])
    assert set(out) == {"PX:SPY", "PX:AGG"} and len(out["PX:SPY"]) == 300

    def down(url, timeout=30):
        raise ConnectionError("blocked")
    monkeypatch.setattr(md, "_get", down)
    out = md.fetch_prices(["SPY", "AGG", "GLD", "DBC", "BIL"])
    assert all(v.empty for v in out.values()) and len(out) == 5     # everything down: empty, not an exception


def test_yahoo_frame_shapes(monkeypatch):
    idx = pd.bdate_range("2024-01-01", periods=300)
    multi = pd.DataFrame({("Close", "SPY"): np.linspace(1, 2, 300), ("Close", "AGG"): np.linspace(3, 4, 300),
                          ("Volume", "SPY"): 1.0, ("Volume", "AGG"): 1.0}, index=idx.tz_localize("America/New_York"))
    multi.columns = pd.MultiIndex.from_tuples(multi.columns)

    class FakeYF:
        @staticmethod
        def download(tickers, **kw):
            return multi

    monkeypatch.setitem(sys.modules, "yfinance", FakeYF)
    out = md._prices_yahoo(["SPY", "AGG"])
    assert set(out) == {"SPY", "AGG"}
    assert out["SPY"].index.tz is None and out["AGG"].iloc[-1] == pytest.approx(4.0)


def test_fetch_all_survives_every_source_failing(monkeypatch):
    def down(*a, **k):
        raise ConnectionError("no network")
    monkeypatch.setattr(md, "_get", down)
    monkeypatch.setattr(md, "_prices_yahoo", down)
    out = md.fetch_all()
    assert out["SHILLER:CAPE"].empty and out["AAII:SPREAD"].empty and out["PX:SPY"].empty


def test_fixture_file_names_map_back_to_keys(tmp_path):
    for name in ("PX_SPY", "SHILLER_CAPE", "AAII_SPREAD", "DGS10"):
        pd.DataFrame({"date": ["2026-01-02"], "value": [1.0]}).to_csv(tmp_path / f"{name}.csv", index=False)
    raw = fred_data.load_fixture(str(tmp_path))
    assert not raw["PX:SPY"].empty and not raw["SHILLER:CAPE"].empty and not raw["AAII:SPREAD"].empty
    assert not raw["DGS10"].empty and raw["DGS2"].empty


def test_tickers_match_what_the_frameworks_read():
    import frameworks as fw
    assert set(fw.PRICE_TICKERS) == set(md.TICKERS)
    assert set(fw.SECTORS) == set(md.SECTORS) and fw.CYCLICAL == md.CYCLICAL
