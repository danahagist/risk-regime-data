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


def all_votes(value, **overrides):
    """Every driver framework votes `value`; alarms are clear and valuation is fair unless overridden."""
    row = {f.id: float(value) for f in fw.FRAMEWORKS}
    row.update({a: 1.0 for a in ALARM_IDS})
    row["buffett"] = 0.0
    row.update(overrides)
    return [row[f.id] for f in fw.FRAMEWORKS]


ALARMS_TRIGGERED = {a: -1.0 for a in ALARM_IDS}


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
    expensive = fw.consensus(votes_frame([all_votes(1, buffett=-1)] * 3)).iloc[-1]
    fair = fw.consensus(votes_frame([all_votes(1)] * 3)).iloc[-1]
    assert expensive["composite"] == pytest.approx(fair["composite"])   # no effect on the score
    assert fair["equity_offset"] == fw.STEP
    assert expensive["equity_offset"] == fw.VALUATION_STEP              # smaller move when expensive


def test_low_agreement_stays_neutral():
    # two driver families strongly positive, two strongly negative: no clear direction
    split = all_votes(1, fed_direction=-1, taylor_gap=-1, fin_conditions=-1, seasonality=-1)
    c = fw.consensus(votes_frame([split] * 3))
    assert c["stance"].map(fw.stance_name).iloc[-1] == "Neutral"


def test_families_are_averaged_before_the_composite():
    # the trend family has three members; it must not count three times
    only_trend = all_votes(0, trend_10m=1, golden_cross=1, abs_momentum=1)
    c = fw.consensus(votes_frame([only_trend])).iloc[-1]
    w = {k: v[1] for k, v in fw.FAMILIES.items() if v[2] == "driver"}
    assert c["composite"] == pytest.approx(w["trend"] / sum(w.values()))


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
