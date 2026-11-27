"""Tests for the original monthly script, run against synthetic series."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import compute_macro_monthly as cmm  # noqa: E402
import make_fixture  # noqa: E402


@pytest.fixture()
def run(tmp_path, monkeypatch):
    def _run(hy_override=None):
        raw = make_fixture.build("2026-11-27")
        if hy_override is not None:
            raw["BAMLH0A0HYM2"] = hy_override(raw["BAMLH0A0HYM2"])
        q = pd.date_range("2019-01-01", "2026-07-01", freq="QS")
        raw["A191RL1Q225SBEA"] = pd.Series(2.5, index=q)
        m = pd.date_range("2019-01-01", "2026-10-01", freq="MS")
        raw["UMCSENT"] = pd.Series(70.0, index=m)
        raw["HOUST"] = pd.Series(1400.0, index=m)

        def fake(code, start):
            return raw[code].loc[start:]

        monkeypatch.setattr(cmm, "fred_series", fake)
        monkeypatch.setattr(cmm, "OUT_PATH", str(tmp_path / "out.csv"))
        monkeypatch.setattr(cmm, "INTERP_REPORT_PATH", str(tmp_path / "interp.csv"))
        monkeypatch.setattr(pd.Timestamp, "today", classmethod(lambda cls, *a, **k: pd.Timestamp("2026-12-02")))
        cmm.main()
        return pd.read_csv(tmp_path / "out.csv")
    return _run


def test_no_partial_month_row(run):
    df = run()
    assert df["date"].iloc[-1] == "2026-11-01"       # a Dec 2 run ends with November
    assert df["date"].iloc[-2] == "2026-10-01"
    assert df["sp500"].iloc[-1] != df["sp500"].iloc[-2]   # not a carry-forward of October


def test_sp500_is_month_end_close(run):
    df = run()
    close = make_fixture.build("2026-11-27")["SP500"].iloc[-1]
    assert df["sp500"].iloc[-1] == pytest.approx(close)


def test_spread_history_is_not_backfilled(run):
    df = run()
    early = df[df["date"] < "2023-01-01"]
    assert early["hy_oas"].isna().all()
    assert (early["hy_oas_available"] == 0).all()
    assert df["hy_oas_available"].iloc[-1] == 1


def test_credit_stress_can_fire_and_reach_off(run):
    # push the spread to 9% for the final three months, as in a credit event
    def spike(s):
        s = s.copy()
        s.loc["2026-09-01":] = 9.0
        return s
    df = run(spike)
    assert df["credit_off"].iloc[-1] == 1
    calm = run()
    assert calm["credit_off"].iloc[-1] == 0
