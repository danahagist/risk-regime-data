import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import backtest as bt  # noqa: E402
import compute_signals  # noqa: E402
import frameworks as fw  # noqa: E402
import make_fixture  # noqa: E402

IDX = pd.date_range("2015-01-31", periods=60, freq="ME")


def rets(stocks, bonds=0.0, cash=0.0, alts=0.0):
    return pd.DataFrame({"stocks": stocks, "bonds": bonds, "cash": cash, "alternatives": alts}, index=IDX)


def test_weights_are_held_the_following_month_not_the_same_one():
    # a signal that "knows" each month's return in advance would show up as a perfect record
    r = pd.Series(np.where(np.arange(60) % 2 == 0, 0.05, -0.05), index=IDX)
    cheat = pd.Series(np.where(r > 0, 1.0, -1.0), index=IDX)       # vote equals the same month's direction
    out = bt.run(bt.follow_weights(cheat), rets(r), cost=0.0)
    # held one month late, it is in stocks exactly for the down months
    assert (out <= 0).all() and out.sum() < -1.0


def test_follow_rule_and_costs():
    vote = pd.Series([1.0] * 30 + [-1.0] * 30, index=IDX)
    w = bt.follow_weights(vote)
    assert w.iloc[0].to_dict() == {"stocks": 1.0, "bonds": 0.0, "cash": 0.0, "alternatives": 0.0}
    assert w.iloc[-1]["cash"] == 1.0
    mixed = bt.follow_weights(pd.Series([0.0], index=IDX[:1])).iloc[0]
    assert mixed["stocks"] == 0.6 and mixed["bonds"] == 0.4
    free = bt.run(w, rets(0.01, cash=0.002), cost=0.0)
    paid = bt.run(w, rets(0.01, cash=0.002), cost=0.001)
    assert free.iloc[:29].eq(0.01).all() and free.iloc[-1] == pytest.approx(0.002)
    assert (free - paid).sum() == pytest.approx(0.001)             # one full switch, charged once


def test_missing_votes_are_not_traded():
    vote = pd.Series([np.nan] * 24 + [1.0] * 36, index=IDX)
    out = bt.run(bt.follow_weights(vote), rets(0.01), cost=0.0)
    assert out.index[0] == IDX[25] and len(out) == 35              # first held month follows the first vote


def test_stats_on_a_known_series():
    r = pd.Series(0.01, index=IDX)
    s = bt.stats(r, pd.Series(0.0, index=IDX))
    assert s["months"] == 60
    assert s["cagr"] == pytest.approx(1.01 ** 12 - 1)
    assert s["max_drawdown"] == pytest.approx(0.0)
    assert s["horizons"]["3m"] == pytest.approx(1.01 ** 3 - 1)
    assert s["horizons"]["1y"] == pytest.approx(1.01 ** 12 - 1)
    assert s["horizons"]["3y"] == pytest.approx(1.01 ** 12 - 1)    # multi-year horizons are annualized
    assert s["horizons"]["10y"] is None                            # not enough history to say
    drop = r.copy()
    drop.iloc[30] = -0.5
    assert bt.stats(drop)["max_drawdown"] == pytest.approx(-0.5)
    assert bt.stats(r.iloc[:20]) is None                           # too short to report


def test_forward_returns_by_vote():
    stock = pd.Series(np.where(np.arange(60) < 30, 0.02, -0.01), index=IDX)
    vote = pd.Series(np.where(np.arange(60) < 20, 1.0, -1.0), index=IDX)
    f = bt.forward_by_vote(vote, stock)
    assert f["1"]["months"] == 20 and f["-1"]["months"] == 40 and f["0"]["months"] == 0
    assert f["1"]["3m"] == pytest.approx(1.02 ** 3 - 1)            # the three months after each positive vote
    assert f["0"]["3m"] is None
    assert f["-1"]["3m"] < f["1"]["3m"]


def test_alt_tilt_series_matches_single_month_rule():
    raw = make_fixture.build("2026-11-27")
    m = fw.build_monthly(raw, asof="2026-11-30")
    series = fw.alt_tilt_series(m)
    for end in ("2026-11-30", "2024-06-30", "2020-04-30", "2012-03-31"):
        assert series[end] == fw.alt_tilt(m.loc[:end])[0], end


def test_sp500_history_is_extended_with_the_fund(tmp_path):
    days = pd.bdate_range("2010-01-01", "2020-12-31")
    spy = pd.Series(np.linspace(100, 300, len(days)), index=days)
    spx = pd.Series(np.linspace(2000, 3700, 1200), index=days[-1200:])
    out = fw.splice_sp500({"SP500": spx, "PX:SPY": spy})
    assert out.index[0] == days[0] and out.loc[spx.index[0]:].equals(spx)
    join = out.loc[:spx.index[0]].iloc[-2:]
    assert abs(join.iloc[1] / join.iloc[0] - 1) < 0.01             # no jump where the two meet
    assert fw.splice_sp500({"SP500": spx}).equals(spx)


def test_full_backtest_on_sample_data_is_valid_json():
    raw = make_fixture.build("2026-11-27")
    out = compute_signals.build_backtest(raw, asof="2026-11-30")
    json.dumps(out, allow_nan=False)
    ids = [p["id"] for p in out["portfolios"]]
    assert ids == ["consensus", "neutral", "sixty_forty", "stocks"]
    assert len({(p["start"], p["end"]) for p in out["portfolios"]}) == 1      # all compared over one window
    assert out["curve"] and set(out["curve"][0]) == {"date", "consensus", "neutral", "sixty_forty", "stocks"}
    tested = [f for f in out["frameworks"] if f["stats"]]
    assert len(tested) >= 20
    for f in tested:
        assert -1 < f["stats"]["max_drawdown"] <= 0
        assert abs(sum(f["time"].values()) - 1) < 0.05
        assert f["benchmarks"]["stocks"]["cagr"] is not None
    # the consensus model moves at most a few points from neutral, so it must track it closely
    c, n = out["portfolios"][0], out["portfolios"][1]
    assert abs(c["cagr"] - n["cagr"]) < 0.03
