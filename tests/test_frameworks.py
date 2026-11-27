import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import frameworks as fw  # noqa: E402
import compute_signals  # noqa: E402
import make_fixture  # noqa: E402

ASOF = "2026-11-30"


@pytest.fixture(scope="module")
def raw():
    return make_fixture.build("2026-11-27")


@pytest.fixture(scope="module")
def monthly(raw):
    return fw.build_monthly(raw, asof=ASOF)


def frame(n=30, **cols):
    """A small monthly frame with every column present, for targeted vote tests."""
    idx = pd.date_range("2024-01-31", periods=n, freq="ME")
    base = fw.build_monthly({}, asof=idx[-1], start=str(idx[0].date()))
    for k, v in cols.items():
        base[k] = v
    return base


# ---------------------------------------------------------------- calendar
def test_last_complete_month():
    assert fw.last_complete_month(pd.Timestamp("2026-12-02")) == pd.Timestamp("2026-11-30")
    assert fw.last_complete_month(pd.Timestamp("2026-11-27")) == pd.Timestamp("2026-10-31")
    assert fw.last_complete_month(pd.Timestamp("2026-11-30")) == pd.Timestamp("2026-11-30")


def test_frame_ends_on_requested_month(monthly):
    assert monthly.index[-1] == pd.Timestamp(ASOF)


# ---------------------------------------------------------------- the three original bugs
def test_credit_signal_uses_percent_units():
    calm = frame(hy_oas=3.2, hy_oas_sma12=3.1)
    stress = frame(hy_oas=6.4, hy_oas_sma12=4.0)
    elevated_but_falling = frame(hy_oas=6.4, hy_oas_sma12=7.5)
    assert fw.v_credit(calm).iloc[-1] == 1
    assert fw.v_credit(stress).iloc[-1] == -1          # could never fire against a 500 threshold
    assert fw.v_credit(elevated_but_falling).iloc[-1] == 0


def test_no_invented_spread_history(monthly):
    # the ICE series only goes back about three years; earlier months must be empty,
    # not padded with the first available value
    early = monthly.loc["2015-01-31":"2022-12-31", "hy_oas"]
    assert early.isna().all()
    votes = fw.compute_votes(monthly)
    assert votes.loc["2015-01-31":"2022-12-31", "hy_spread"].notna().all()  # falls back to the Fed credit subindex


def test_trend_uses_month_end_close(raw, monthly):
    last_close = raw["SP500"].loc[:ASOF].iloc[-1]
    assert monthly["sp500"].iloc[-1] == pytest.approx(last_close)
    median = raw["SP500"].loc["2026-11-01":ASOF].median()
    assert abs(monthly["sp500"].iloc[-1] - median) > 1  # and it is not the monthly median


# ---------------------------------------------------------------- individual frameworks
def test_yield_curve_probability_matches_published_formula():
    m = frame(curve=1.0)
    assert fw.curve_prob(m).iloc[-1] == pytest.approx(12.2, abs=0.2)
    assert fw.v_curve(m).iloc[-1] == 1
    assert fw.v_curve(frame(curve=-1.0)).iloc[-1] == -1      # inverted: about 54%


def test_sahm_thresholds():
    assert fw.v_sahm(frame(sahm=0.10)).iloc[-1] == 1
    assert fw.v_sahm(frame(sahm=0.40)).iloc[-1] == 0
    assert fw.v_sahm(frame(sahm=0.50)).iloc[-1] == -1


def test_quadrants():
    assert fw.v_quadrant(frame(growth_chg=0.2, infl_chg=-0.2)).iloc[-1] == 1     # Goldilocks
    assert fw.v_quadrant(frame(growth_chg=-0.2, infl_chg=0.2)).iloc[-1] == 0     # Stagflation
    assert fw.v_quadrant(frame(growth_chg=-0.2, infl_chg=-0.2)).iloc[-1] == -1   # Deflation


def test_taylor_loose_policy_only_helps_when_inflation_is_contained():
    loose_low_infl = frame(taylor_gap=-2.0, core_pce_yoy=2.4)
    loose_high_infl = frame(taylor_gap=-2.0, core_pce_yoy=3.4)
    tight = frame(taylor_gap=1.5, core_pce_yoy=2.4)
    assert fw.v_taylor(loose_low_infl).iloc[-1] == 1
    assert fw.v_taylor(loose_high_infl).iloc[-1] == 0
    assert fw.v_taylor(tight).iloc[-1] == -1


def test_trend_is_binary():
    assert fw.v_trend(frame(sp500=100.0, sp500_sma10=99.0)).iloc[-1] == 1
    assert fw.v_trend(frame(sp500=98.0, sp500_sma10=99.0)).iloc[-1] == -1


def test_seasonality_looks_at_the_coming_month():
    idx = pd.DatetimeIndex(["2026-11-30", "2026-05-31", "2028-05-31"])
    best_six, cycle = fw.seasonality_parts(idx)
    assert list(best_six) == [1.0, 0.0, 0.0]      # Dec yes, Jun no, Jun no
    assert list(cycle) == [1.0, -1.0, 0.0]        # post-midterm, weak midterm stretch, election year


def test_missing_data_gives_no_vote_not_a_crash():
    votes = fw.compute_votes(frame())
    assert votes.drop(columns=["seasonality"]).iloc[-1].isna().all()
    cons = fw.consensus(votes)
    assert np.isnan(cons["composite"].iloc[-1])   # one driver family is not enough to take a stance


# ---------------------------------------------------------------- consensus rules
def votes_frame(rows):
    idx = pd.date_range("2025-01-31", periods=len(rows), freq="ME")
    cols = [f.id for f in fw.FRAMEWORKS]
    return pd.DataFrame(rows, index=idx, columns=cols, dtype="float64")


ALARM_IDS = [f.id for f in fw.FRAMEWORKS if fw.FAMILIES[f.family][2] == "alarm"]
VALUATION_IDS = [f.id for f in fw.FRAMEWORKS if f.family == "valuation"]
POLICY_IDS = [f.id for f in fw.FRAMEWORKS if f.family == "policy"]
TREND_IDS = [f.id for f in fw.FRAMEWORKS if f.family == "trend"]
CALENDAR_IDS = [f.id for f in fw.FRAMEWORKS if f.family == "calendar"]


def all_votes(value, **overrides):
    """Every driver framework votes `value`; alarms are clear and valuation is fair unless overridden."""
    row = {f.id: float(value) for f in fw.FRAMEWORKS}
    row.update({a: 1.0 for a in ALARM_IDS})
    row.update({v: 0.0 for v in VALUATION_IDS})
    row.update(overrides)
    return [row[f.id] for f in fw.FRAMEWORKS]


ALARMS_TRIGGERED = {a: -1.0 for a in ALARM_IDS}
EXPENSIVE = {v: -1.0 for v in [f.id for f in fw.FRAMEWORKS if f.family == "valuation"]}


def test_change_needs_two_consecutive_readings():
    v = votes_frame([all_votes(0), all_votes(1), all_votes(1)])
    stance = fw.consensus(v)["stance"].map(fw.stance_name).tolist()
    assert stance == ["Neutral", "Neutral", "Overweight"]


def test_one_step_at_a_time():
    v = votes_frame([all_votes(1)] * 3 + [all_votes(-1)] * 4)
    stance = fw.consensus(v)["stance"].map(fw.stance_name).tolist()
    assert stance == ["Neutral", "Overweight", "Overweight",
                      "Overweight", "Neutral", "Underweight", "Underweight"]


def test_alarms_can_only_subtract():
    clear = fw.consensus(votes_frame([all_votes(0)])).iloc[-1]
    assert clear["fam_recession"] == 0 and clear["fam_credit_vol"] == 0
    assert clear["composite"] == pytest.approx(0.0)        # clear alarms add nothing
    watch = fw.consensus(votes_frame([all_votes(0, **{a: 0.0 for a in ALARM_IDS})])).iloc[-1]
    assert watch["composite"] == pytest.approx(-0.5)
    ringing = fw.consensus(votes_frame([all_votes(0, **ALARMS_TRIGGERED)])).iloc[-1]
    assert ringing["composite"] == pytest.approx(-1.0)


def test_breaker_acts_at_once_then_underweight_needs_confirmation():
    # drivers stay fully positive, then every alarm triggers
    v = votes_frame([all_votes(1)] * 3 + [all_votes(1, **ALARMS_TRIGGERED)] * 3)
    c = fw.consensus(v)
    assert bool(c["breaker_recession"].iloc[-1]) and bool(c["breaker_credit"].iloc[-1])
    stance = c["stance"].map(fw.stance_name).tolist()
    assert stance[:4] == ["Neutral", "Overweight", "Overweight", "Neutral"]   # cut at once
    assert "Overweight" not in stance[3:]


def test_recession_breaker_caps_equity():
    recession = all_votes(1, yield_curve=-1, sahm=-1, cfnai_ma3=-1, claims=-1)
    c = fw.consensus(votes_frame([recession] * 3))
    assert bool(c["breaker_recession"].iloc[-1])
    assert c["stance"].map(fw.stance_name).tolist() == ["Neutral"] * 3


def test_valuation_sizes_but_does_not_time():
    expensive = fw.consensus(votes_frame([all_votes(1, **EXPENSIVE)] * 3)).iloc[-1]
    fair = fw.consensus(votes_frame([all_votes(1)] * 3)).iloc[-1]
    assert expensive["composite"] == pytest.approx(fair["composite"])   # no effect on the score
    assert fair["equity_offset"] == fw.STEP
    assert expensive["equity_offset"] == fw.VALUATION_STEP              # smaller move when expensive


def test_low_agreement_stays_neutral():
    # two driver families strongly positive, two strongly negative: no clear direction
    split = all_votes(1, **{k: -1.0 for k in POLICY_IDS + CALENDAR_IDS})
    c = fw.consensus(votes_frame([split] * 3))
    assert c["stance"].map(fw.stance_name).iloc[-1] == "Neutral"


def test_families_are_averaged_before_the_composite():
    # the trend family has several members; it must count once
    only_trend = all_votes(0, **{k: 1.0 for k in TREND_IDS})
    c = fw.consensus(votes_frame([only_trend])).iloc[-1]
    w = {k: v[1] for k, v in fw.FAMILIES.items() if v[2] == "driver"}
    assert c["composite"] == pytest.approx(w["trend"] / sum(w.values()))


def test_info_frameworks_never_move_the_score():
    info = {f.id: 1.0 for f in fw.FRAMEWORKS if fw.FAMILIES[f.family][2] == "info"}
    assert info, "expected at least one info framework"
    up = fw.consensus(votes_frame([all_votes(0, **info)])).iloc[-1]
    down = fw.consensus(votes_frame([all_votes(0, **{k: -1.0 for k in info})])).iloc[-1]
    assert up["composite"] == pytest.approx(down["composite"]) == pytest.approx(0.0)


# ---------------------------------------------------------------- price, valuation and liquidity frameworks
def test_dual_momentum_follows_antonacci():
    def hold(us, intl, bills):
        return fw.dual_momentum_holding(frame(ret12_SPY=us, ret12_ACWX=intl, ret12_BIL=bills)).iloc[-1]
    assert hold(12.0, 8.0, 4.0) == "US stocks"
    assert hold(12.0, 15.0, 4.0) == "Foreign stocks"
    assert hold(2.0, 9.0, 4.0) == "Bonds"          # the test is US stocks against T-bills
    assert fw.v_dual_momentum(frame(ret12_SPY=2.0, ret12_ACWX=9.0, ret12_BIL=4.0)).iloc[-1] == -1


def test_breadth_thresholds():
    assert fw.v_breadth(frame(breadth=64.0)).iloc[-1] == 1
    assert fw.v_breadth(frame(breadth=50.0)).iloc[-1] == 0
    assert fw.v_breadth(frame(breadth=36.0)).iloc[-1] == -1


def test_breadth_counts_sectors_above_average(monthly):
    r = monthly.iloc[-1]
    above = sum(r[f"px_{t}"] >= r[f"sma_{t}"] for t in fw.SECTORS)
    assert r["breadth"] == pytest.approx(above / len(fw.SECTORS) * 100)


def test_fed_model_and_rule_of_20():
    assert fw.v_fed_model(frame(fed_model_gap=2.5)).iloc[-1] == 1
    assert fw.v_fed_model(frame(fed_model_gap=0.8)).iloc[-1] == 0
    assert fw.v_fed_model(frame(fed_model_gap=-0.4)).iloc[-1] == -1
    assert fw.v_rule20(frame(rule20=17.0)).iloc[-1] == 1
    assert fw.v_rule20(frame(rule20=25.0)).iloc[-1] == -1


def test_stale_valuation_history_is_estimated_forward(raw, monthly):
    # the published CAPE stops in 2023; recent months must be estimated, flagged, and joined without a jump
    published = raw["SHILLER:CAPE"]
    assert published.index[-1] < pd.Timestamp("2024-01-01")
    last = monthly.iloc[-1]
    assert bool(last["valuation_estimated"]) and 15 < last["cape"] < 60 and 10 < last["pe"] < 45
    join = published.index[-1] + pd.offsets.MonthEnd(0)
    assert monthly.loc[join, "cape"] == pytest.approx(published.iloc[-1])
    step = monthly["cape"].loc[join:].pct_change().abs().iloc[1]
    assert step < 0.10                                   # chain-linked: no break at the join
    assert not monthly.loc[join, "valuation_estimated"]
    # the estimated P/E matches price over interpolated earnings at the latest month
    assert last["pe"] == pytest.approx(monthly["sp500"].iloc[-1] / 292.0, rel=0.08)


def test_earnings_point_observed_after_the_month_still_anchors_it(raw):
    # the job runs on the 2nd, so the P/E it observes is dated in the month after the one reported
    m = fw.build_monthly(raw, asof="2026-10-31")       # the fixture's point is dated 2026-11-27
    last = m.iloc[-1]
    assert bool(last["valuation_estimated"]) and 15 < last["cape"] < 60 and 10 < last["pe"] < 45
    assert fw.v_cape(m).iloc[-1] in (-1, 0, 1)


def test_fresh_valuation_history_is_used_as_published(raw):
    fresh = dict(raw)
    idx = pd.date_range("1900-01-31", "2026-11-30", freq="ME")
    fresh["SHILLER:CAPE"] = pd.Series(np.linspace(15, 38, len(idx)), index=idx)
    fresh["SHILLER:PE"] = pd.Series(np.linspace(14, 25, len(idx)), index=idx)
    m = fw.build_monthly(fresh, asof=ASOF)
    assert m["cape"].iloc[-1] == pytest.approx(38.0) and m["pe"].iloc[-1] == pytest.approx(25.0)
    assert not m["valuation_estimated"].any()


def test_valuation_without_a_current_earnings_point_reports_no_data(raw):
    stale = {k: v for k, v in raw.items() if k != "VAL:EPS_TTM"}
    m = fw.build_monthly(stale, asof=ASOF)
    assert np.isnan(m["cape"].iloc[-1]) and np.isnan(m["pe"].iloc[-1])   # three years old is not a reading
    assert np.isnan(fw.v_cape(m).iloc[-1])


def test_sentiment_is_contrarian():
    assert fw.v_sentiment(frame(aaii_spread=-30.0)).iloc[-1] == 1     # deep pessimism
    assert fw.v_sentiment(frame(aaii_spread=5.0)).iloc[-1] == 0
    assert fw.v_sentiment(frame(aaii_spread=35.0)).iloc[-1] == -1     # euphoria


def test_net_liquidity_handles_mixed_units(monthly, raw):
    # WALCL arrives in millions, the other two in billions
    assert raw["WALCL"].iloc[-1] > 1_000_000 and raw["WTREGEN"].iloc[-1] < 5_000
    last = monthly["net_liquidity"].iloc[-1]
    assert 3_000 < last < 9_000          # billions of dollars
    assert fw._to_billions(pd.Series([6_500_000.0, 6_600_000.0])).iloc[-1] == pytest.approx(6_600.0)
    assert fw._to_billions(pd.Series([800.0, 850.0])).iloc[-1] == pytest.approx(850.0)


def test_alternatives_tilt_needs_an_uptrend():
    rising = dict(infl_chg=0.3, growth_chg=0.1)
    up = frame(**rising, px_GLD=110.0, sma_GLD=100.0, px_DBC=90.0, sma_DBC=100.0)
    down = frame(**rising, px_GLD=90.0, sma_GLD=100.0, px_DBC=90.0, sma_DBC=100.0)
    unknown = frame(**rising)
    assert fw.alt_tilt(up)[0] == fw.ALT_STEP
    assert fw.alt_tilt(down)[0] == 0                       # inflation up but nothing trending
    assert fw.alt_tilt(unknown)[0] == fw.ALT_STEP          # no price data: fall back to the inflation rule
    assert fw.alt_tilt(frame(infl_chg=-0.3, growth_chg=-0.1))[0] == -fw.ALT_STEP


def test_fred_only_run_still_works(raw):
    # outside sources can all fail on the same day; the FRED frameworks must carry on
    fred_only = {k: v for k, v in raw.items() if ":" not in k}
    payload, _ = compute_signals.build_payload(fred_only, asof=ASOF)
    by_id = {f["id"]: f for f in payload["frameworks"]}
    for fid in ("dual_momentum", "breadth", "dow_theory", "cape", "fed_model", "rule_of_20", "sentiment",
                "gtaa", "sector_rotation", "intermarket"):
        assert by_id[fid]["vote"] is None, fid
    assert by_id["trend_10m"]["vote"] is not None and by_id["net_liquidity"]["vote"] is not None
    assert payload["stance"]["label"] in fw.STANCES
    assert sum(payload["allocation"]["model"].values()) == 100
    json.dumps(payload, allow_nan=False)


def test_tables_are_plain_json(raw):
    payload, _ = compute_signals.build_payload(raw, asof=ASOF)
    by_id = {f["id"]: f for f in payload["frameworks"]}
    sectors = by_id["sector_rotation"]["table"]
    assert len(sectors["rows"]) == len(fw.SECTORS) and len(sectors["columns"]) == len(sectors["rows"][0])
    assert [r[0] for r in sectors["rows"]] == list(range(1, len(fw.SECTORS) + 1))
    assert len(by_id["gtaa"]["table"]["rows"]) == len(fw.TREND_ASSETS)
    json.dumps(payload, allow_nan=False)


# ---------------------------------------------------------------- allocation
def test_model_allocation_always_sums_to_100():
    for offset in (-10, -5, 0, 5, 10):
        for tilt in (-5, 0, 5):
            for defensive in ("cash", "bonds", "split"):
                for neutral in (fw.NEUTRAL_MIX, {"stocks": 80, "bonds": 20, "cash": 0, "alternatives": 0},
                                {"stocks": 100, "bonds": 0, "cash": 0, "alternatives": 0}):
                    mix = fw.model_allocation(neutral, offset, tilt, defensive)
                    assert sum(mix.values()) == 100
                    assert all(v >= 0 for v in mix.values())


def test_model_allocation_rules():
    n = fw.NEUTRAL_MIX
    assert fw.model_allocation(n, 0, 0, "split") == n                       # nothing to say: hold neutral
    hiking = fw.model_allocation(n, 0, 0, "cash")
    cutting = fw.model_allocation(n, 0, 0, "bonds")
    assert hiking["cash"] > n["cash"] > cutting["cash"]                    # cash while hiking, bonds once cutting
    assert hiking["stocks"] == cutting["stocks"] == n["stocks"]            # policy does not move stocks
    assert fw.model_allocation(n, 10, 0, "split")["stocks"] == 70
    assert fw.model_allocation(n, 0, 5, "split")["alternatives"] == 10


def test_asset_views_reference_real_frameworks(raw):
    payload, _ = compute_signals.build_payload(raw, asof=ASOF)
    ids = {f["id"] for f in payload["frameworks"]}
    a = payload["allocation"]
    assert a["views"] and all(v["id"] in ids for v in a["views"])
    assert all(set(v["favors"]) <= set(fw.ASSETS) for v in a["views"])
    assert sum(a["model"].values()) == 100


def test_every_live_framework_and_metric_has_a_lesson(raw):
    lessons = json.load(open(os.path.join(ROOT, "docs", "data", "lessons.json")))
    live = {x["id"] for x in lessons["lessons"] if x["status"] == "live"}
    assert live == {f.id for f in fw.FRAMEWORKS}
    payload, _ = compute_signals.build_payload(raw, asof=ASOF)
    missing = [x["key"] for x in payload["metrics"] if x["key"] not in lessons["metrics"]]
    assert not missing, missing


# ---------------------------------------------------------------- end to end
def test_payload_is_valid_json_with_no_nan(raw):
    payload, full = compute_signals.build_payload(raw, asof=ASOF)
    text = json.dumps(payload, allow_nan=False)        # raises if any NaN slipped through
    back = json.loads(text)
    assert back["as_of"] == ASOF
    assert len(back["frameworks"]) == len(fw.FRAMEWORKS)
    assert all(f["vote"] in (-1, 0, 1) for f in back["frameworks"])
    assert back["stance"]["label"] in fw.STANCES
    assert {f["id"] for f in back["families"]} == set(fw.FAMILIES)
    assert len(back["history"]) > 24
    assert not full.empty


def test_run_survives_missing_series(raw):
    partial = {k: v for k, v in raw.items() if k not in ("SP500", "BAMLH0A0HYM2", "VXVCLS", "NROU")}
    payload, _ = compute_signals.build_payload(partial, asof=ASOF)
    by_id = {f["id"]: f for f in payload["frameworks"]}
    assert by_id["trend_10m"]["vote"] is None
    assert by_id["trend_10m"]["vote_label"] == "No data"
    assert by_id["yield_curve"]["vote"] is not None
    json.dumps(payload, allow_nan=False)
