"""Framework votes and consensus scoring.

Pure functions, no network access. Everything is evaluated at month-end on the
last complete month, using only data that would have been published by then.

Vote convention: +1 favors holding more stocks, -1 favors less, 0 is neutral,
NaN means the framework has no data for that month.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

# ----------------------------------------------------------------------------
# Parameters (set once, on purpose; do not tune these to a backtest)
# ----------------------------------------------------------------------------
R_STAR = 1.0                 # neutral real rate used in the Taylor rule
INFLATION_TARGET = 2.0
STANCE_THRESHOLD = 0.5       # composite needed to leave neutral
MIN_AGREEMENT = 0.5          # share of opinionated families that must agree
STEP = 10                    # equity points per stance step
VALUATION_STEP = 5           # reduced step when valuation argues against the move

# Three kinds of family:
#   driver  votes on direction, from -1 to +1
#   alarm   can only subtract. An alarm that is not ringing is the absence of bad
#           news, not good news, so its score runs from -1 (triggered) to 0 (clear)
#   sizing  never moves the stance; it sets how large a move is allowed to be
FAMILIES = {
    # id: (name, weight, role)
    "macro": ("Macro regime", 0.75, "driver"),
    "policy": ("Policy and liquidity", 0.75, "driver"),
    "trend": ("Trend and momentum", 1.0, "driver"),
    "calendar": ("Sentiment and calendar", 0.5, "driver"),
    "recession": ("Recession risk", 1.0, "alarm"),
    "credit_vol": ("Credit and volatility", 1.0, "alarm"),
    "valuation": ("Valuation", 0.75, "sizing"),
}
MIN_DRIVERS = 2              # driver families needed to say anything
BREAKER_LEVEL = -0.75        # alarm family score that caps equity at Neutral

STANCES = ["Underweight", "Neutral", "Overweight"]


# ----------------------------------------------------------------------------
# Monthly data frame
# ----------------------------------------------------------------------------
def last_complete_month(today: pd.Timestamp | None = None) -> pd.Timestamp:
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today).normalize()
    this_month_end = today + pd.offsets.MonthEnd(0)
    return this_month_end if today == this_month_end else today - pd.offsets.MonthEnd(1)


def _daily(raw, code, idx, how="last"):
    s = raw.get(code, pd.Series(dtype="float64")).dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=idx)
    r = s.resample("ME")
    m = r.last() if how == "last" else r.mean()
    return m.reindex(idx)


def _monthly(raw, code, idx, lag=1, ffill=3):
    """Monthly series, shifted by its publication lag so a month only sees data
    that was out by that month-end."""
    s = raw.get(code, pd.Series(dtype="float64")).dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=idx)
    s = s.copy()
    s.index = s.index.to_period("M").to_timestamp("M") + pd.offsets.MonthEnd(lag)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(idx).ffill(limit=ffill)


def _quarterly(raw, code, idx, lag=3, ffill=6):
    s = raw.get(code, pd.Series(dtype="float64")).dropna().sort_index()
    if s.empty:
        return pd.Series(np.nan, index=idx)
    s = s.copy()
    s.index = s.index.to_period("Q").to_timestamp("Q") + pd.offsets.MonthEnd(lag)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(idx).ffill(limit=ffill)


def build_monthly(raw: dict[str, pd.Series], asof: pd.Timestamp | None = None,
                  start: str = "1990-01-31") -> pd.DataFrame:
    """One row per month-end, one column per indicator the frameworks read."""
    asof = last_complete_month() if asof is None else pd.Timestamp(asof) + pd.offsets.MonthEnd(0)
    idx = pd.date_range(start=start, end=asof, freq="ME")
    m = pd.DataFrame(index=idx)

    # growth
    m["cfnai"] = _monthly(raw, "CFNAI", idx)
    m["cfnai_ma3"] = m["cfnai"].rolling(3, min_periods=3).mean()
    m["growth_chg"] = m["cfnai_ma3"] - m["cfnai_ma3"].shift(3)
    indpro = _monthly(raw, "INDPRO", idx)
    m["indpro_yoy"] = indpro.pct_change(12, fill_method=None) * 100

    # inflation
    core = _monthly(raw, "PCEPILFE", idx)
    m["core_pce_yoy"] = core.pct_change(12, fill_method=None) * 100
    m["infl_chg"] = m["core_pce_yoy"] - m["core_pce_yoy"].shift(6)
    cpi = _monthly(raw, "CPIAUCSL", idx)
    m["cpi_yoy"] = cpi.pct_change(12, fill_method=None) * 100

    # labor
    m["unrate"] = _monthly(raw, "UNRATE", idx)
    m["unrate_low36"] = m["unrate"].rolling(36, min_periods=24).min()
    m["sahm"] = _monthly(raw, "SAHMREALTIME", idx)
    claims = raw.get("ICSA", pd.Series(dtype="float64")).dropna().sort_index()
    if claims.empty:
        m["claims_4wk"] = np.nan
    else:
        m["claims_4wk"] = claims.rolling(4, min_periods=4).mean().resample("ME").last().reindex(idx)
    m["claims_yoy"] = m["claims_4wk"].pct_change(12, fill_method=None) * 100

    # rates and policy
    m["y10"] = _daily(raw, "DGS10", idx, "mean")
    m["y2"] = _daily(raw, "DGS2", idx, "mean")
    m["y3m"] = _daily(raw, "DGS3MO", idx, "mean")
    m["curve"] = m["y10"] - m["y3m"]
    target = _daily(raw, "DFEDTARU", idx, "last")
    eff = _monthly(raw, "FEDFUNDS", idx, lag=0, ffill=2)
    m["fed_funds"] = target.where(target.notna(), eff)
    m["fed_chg6"] = m["fed_funds"] - m["fed_funds"].shift(6)
    m["fed_chg12"] = m["fed_funds"] - m["fed_funds"].shift(12)
    nrou = raw.get("NROU", pd.Series(dtype="float64")).dropna().sort_index()
    if nrou.empty:
        m["nrou"] = np.nan
    else:
        nrou = nrou.copy()
        nrou.index = nrou.index.to_period("M").to_timestamp("M")
        m["nrou"] = nrou.reindex(idx.union(nrou.index)).ffill().reindex(idx)
    m["taylor_rate"] = (R_STAR + m["core_pce_yoy"] + 0.5 * (m["core_pce_yoy"] - INFLATION_TARGET)
                        + 0.5 * 2.0 * (m["nrou"] - m["unrate"]))
    m["taylor_gap"] = m["fed_funds"] - m["taylor_rate"]
    m["nfci"] = _daily(raw, "NFCI", idx, "last")
    m["nfci_chg3"] = m["nfci"] - m["nfci"].shift(3)
    m["nfci_credit"] = _daily(raw, "NFCICREDIT", idx, "last")

    # valuation: market value of corporate equities / GDP, ranked against its own past
    eq = raw.get("NCBEILQ027S", pd.Series(dtype="float64")).dropna().sort_index()
    gdp = raw.get("GDP", pd.Series(dtype="float64")).dropna().sort_index()
    if eq.empty or gdp.empty:
        m["buffett"] = np.nan
        m["buffett_pctile"] = np.nan
    else:
        ratio = (eq / (gdp * 1000.0)).dropna() * 100.0  # percent of GDP
        pct = ratio.expanding(min_periods=40).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
        m["buffett"] = _quarterly({"x": ratio}, "x", idx)
        m["buffett_pctile"] = _quarterly({"x": pct}, "x", idx)

    # stocks: month-end close, the convention the 10-month rule is defined on
    spx_d = raw.get("SP500", pd.Series(dtype="float64")).dropna().sort_index()
    m["sp500"] = _daily(raw, "SP500", idx, "last")
    m["sp500_sma10"] = m["sp500"].rolling(10, min_periods=10).mean()
    m["sp500_ret12"] = m["sp500"].pct_change(12, fill_method=None) * 100
    m["bill_12m_ago"] = m["y3m"].shift(12)
    if spx_d.empty:
        m["sp500_sma50d"] = np.nan
        m["sp500_sma200d"] = np.nan
    else:
        m["sp500_sma50d"] = spx_d.rolling(50, min_periods=50).mean().resample("ME").last().reindex(idx)
        m["sp500_sma200d"] = spx_d.rolling(200, min_periods=200).mean().resample("ME").last().reindex(idx)

    # credit and volatility
    m["hy_oas"] = _daily(raw, "BAMLH0A0HYM2", idx, "last")       # percent, as FRED reports it
    m["hy_oas_sma12"] = m["hy_oas"].rolling(12, min_periods=12).mean()
    vix_d = raw.get("VIXCLS", pd.Series(dtype="float64")).dropna().sort_index()
    if vix_d.empty:
        m["vix_avg"] = np.nan
    else:
        m["vix_avg"] = vix_d.rolling(21, min_periods=15).mean().resample("ME").last().reindex(idx)
    m["vix"] = _daily(raw, "VIXCLS", idx, "last")
    m["vix_3m"] = _daily(raw, "VXVCLS", idx, "last")
    m["vix_ratio"] = m["vix"] / m["vix_3m"]
    return m


# ----------------------------------------------------------------------------
# Vote helpers
# ----------------------------------------------------------------------------
def _tri(pos: pd.Series, neg: pd.Series, valid: pd.Series) -> pd.Series:
    out = pd.Series(0.0, index=valid.index)
    out[pos.fillna(False)] = 1.0
    out[neg.fillna(False)] = -1.0
    out[~valid.fillna(False)] = np.nan
    return out


def _ok(*cols: pd.Series) -> pd.Series:
    v = cols[0].notna()
    for c in cols[1:]:
        v &= c.notna()
    return v


# ----------------------------------------------------------------------------
# Frameworks
# ----------------------------------------------------------------------------
def quadrant_label(m):
    g, i = m["growth_chg"] > 0, m["infl_chg"] > 0
    lab = np.select([g & ~i, g & i, ~g & i, ~g & ~i],
                    ["Goldilocks", "Reflation", "Stagflation", "Deflation"], default="")
    return pd.Series(lab, index=m.index).where(_ok(m["growth_chg"], m["infl_chg"]))


def v_quadrant(m):
    q = quadrant_label(m)
    return _tri(q.isin(["Goldilocks", "Reflation"]), q == "Deflation", q.notna())


def clock_label(m):
    strong, rising = m["cfnai_ma3"] > 0, m["infl_chg"] > 0
    lab = np.select([strong & ~rising, strong & rising, ~strong & rising, ~strong & ~rising],
                    ["Recovery", "Overheat", "Stagflation", "Reflation"], default="")
    return pd.Series(lab, index=m.index).where(_ok(m["cfnai_ma3"], m["infl_chg"]))


def v_clock(m):
    c = clock_label(m)
    return _tri(c == "Recovery", c == "Stagflation", c.notna())


def cycle_label(m):
    valid = _ok(m["cfnai_ma3"], m["unrate"], m["unrate_low36"], m["curve"], m["fed_chg12"])
    recession = (m["cfnai_ma3"] < -0.7) | (m["sahm"] >= 0.5)
    slack = m["unrate"] - m["unrate_low36"]
    early = ~recession & (slack >= 1.0) & (m["growth_chg"] > 0)
    late = ~recession & ~early & (slack <= 0.5) & ((m["fed_chg12"] >= 0.5) | (m["curve"] < 0.5))
    lab = np.select([recession, early, late], ["Recession", "Early", "Late"], default="Mid")
    return pd.Series(lab, index=m.index).where(valid)


def v_cycle(m):
    c = cycle_label(m)
    return _tri(c.isin(["Early", "Mid"]), c == "Recession", c.notna())


def curve_prob(m):
    """NY Fed style probit (Estrella and Trubin 2006): chance of a recession 12 months out."""
    return pd.Series(norm.cdf(-0.5333 - 0.6330 * m["curve"]) * 100, index=m.index).where(m["curve"].notna())


def v_curve(m):
    p = curve_prob(m)
    return _tri(p < 20, p >= 40, p.notna())


def v_sahm(m):
    return _tri(m["sahm"] < 0.30, m["sahm"] >= 0.50, m["sahm"].notna())


def v_cfnai(m):
    return _tri(m["cfnai_ma3"] >= 0, m["cfnai_ma3"] < -0.7, m["cfnai_ma3"].notna())


def v_claims(m):
    return _tri(m["claims_yoy"] < 10, m["claims_yoy"] > 20, m["claims_yoy"].notna())


def v_fed(m):
    return _tri(m["fed_chg6"] <= -0.25, m["fed_chg6"] >= 0.25, m["fed_chg6"].notna())


def v_taylor(m):
    # Tight policy is a headwind. Loose policy helps only while inflation is contained;
    # loose with inflation above 3% means the Fed is behind and has more to do.
    loose_ok = (m["taylor_gap"] < -1.0) & (m["core_pce_yoy"] <= 3.0)
    return _tri(loose_ok, m["taylor_gap"] > 1.0, _ok(m["taylor_gap"], m["core_pce_yoy"]))


def v_nfci(m):
    easy = (m["nfci"] <= 0) & (m["nfci_chg3"] <= 0.10)
    return _tri(easy, m["nfci"] > 0, _ok(m["nfci"], m["nfci_chg3"]))


def v_buffett(m):
    return _tri(m["buffett_pctile"] < 40, m["buffett_pctile"] > 80, m["buffett_pctile"].notna())


def v_trend(m):
    up = m["sp500"] >= m["sp500_sma10"]
    return _tri(up, ~up, _ok(m["sp500"], m["sp500_sma10"]))


def v_cross(m):
    up = m["sp500_sma50d"] >= m["sp500_sma200d"]
    return _tri(up, ~up, _ok(m["sp500_sma50d"], m["sp500_sma200d"]))


def v_momentum(m):
    up = m["sp500_ret12"] > m["bill_12m_ago"]
    return _tri(up, ~up, _ok(m["sp500_ret12"], m["bill_12m_ago"]))


def v_credit(m):
    """High-yield spread, in percent. Stress when the spread is above both 5% and its own
    12-month average. Where the ICE series has no history, fall back to the Chicago Fed
    credit subindex so the family still has a vote."""
    barrier = pd.concat([pd.Series(5.0, index=m.index), m["hy_oas_sma12"]], axis=1).max(axis=1)
    calm = (m["hy_oas"] < 4.0) & (m["hy_oas"] <= m["hy_oas_sma12"] * 1.10)
    hy = _tri(calm, m["hy_oas"] > barrier, _ok(m["hy_oas"], m["hy_oas_sma12"]))
    fallback = _tri(m["nfci_credit"] <= 0, m["nfci_credit"] > 0.25, m["nfci_credit"].notna())
    return hy.where(hy.notna(), fallback)


def v_vix(m):
    calm = (m["vix_avg"] < 20) & ~(m["vix_ratio"] > 1.0)
    return _tri(calm, m["vix_avg"] > 30, m["vix_avg"].notna())


def seasonality_parts(idx: pd.DatetimeIndex):
    """Calendar tendencies for the month AFTER each month-end (the month being positioned for)."""
    nxt = idx + pd.offsets.MonthEnd(1)
    month, year = nxt.month, nxt.year
    best_six = pd.Series(((month >= 11) | (month <= 4)).astype(float), index=idx)
    midterm = (year % 4 == 2)
    pre_election = (year % 4 == 3)
    post_midterm = (midterm & (month >= 11)) | pre_election
    weak_midterm = midterm & (month <= 10)
    cycle = pd.Series(np.where(post_midterm, 1.0, np.where(weak_midterm, -1.0, 0.0)), index=idx)
    return best_six, cycle


def v_season(m):
    best_six, cycle = seasonality_parts(m.index)
    total = best_six + cycle
    return _tri(total >= 1, total < 0, pd.Series(True, index=m.index))


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------
def _f(x, digits=2, unit=""):
    return None if x is None or pd.isna(x) else f"{x:,.{digits}f}{unit}"


@dataclass
class Framework:
    id: str
    name: str
    family: str
    weight: float                      # evidence weight: 1.0, 0.75 or 0.5
    rule: str
    vote: Callable[[pd.DataFrame], pd.Series]
    reading: Callable[[pd.Series, pd.DataFrame], str]
    inputs: Callable[[pd.Series], list] = field(default=lambda r: [])


def _inp(label, value, digits=2, unit=""):
    return {"label": label, "value": _f(value, digits, unit)}


FRAMEWORKS: list[Framework] = [
    Framework("quadrant", "Growth and inflation quadrants", "macro", 0.75,
              "Growth and inflation each accelerating or slowing. Goldilocks or Reflation: +1. Stagflation: 0. Deflation: -1.",
              v_quadrant,
              lambda r, m: f"{quadrant_label(m).iloc[-1]}: growth {'accelerating' if r.growth_chg > 0 else 'slowing'}, inflation {'accelerating' if r.infl_chg > 0 else 'slowing'}",
              lambda r: [_inp("CFNAI 3-mo avg, change over 3 months", r.growth_chg),
                         _inp("Core PCE inflation", r.core_pce_yoy, 1, "%"),
                         _inp("Change in inflation over 6 months", r.infl_chg, 2, " pts")]),
    Framework("investment_clock", "Investment Clock", "macro", 0.75,
              "Growth above or below trend, inflation rising or falling. Recovery: +1. Overheat or Reflation: 0. Stagflation: -1.",
              v_clock,
              lambda r, m: f"{clock_label(m).iloc[-1]}: growth {'above' if r.cfnai_ma3 > 0 else 'below'} trend, inflation {'rising' if r.infl_chg > 0 else 'falling'}",
              lambda r: [_inp("CFNAI 3-mo avg (0 = trend growth)", r.cfnai_ma3),
                         _inp("Change in inflation over 6 months", r.infl_chg, 2, " pts")]),
    Framework("business_cycle", "Business cycle phase", "macro", 0.75,
              "Early or Mid cycle: +1. Late cycle: 0. Recession: -1.",
              v_cycle,
              lambda r, m: f"{cycle_label(m).iloc[-1]} cycle",
              lambda r: [_inp("Unemployment above its 3-year low", r.unrate - r.unrate_low36, 1, " pts"),
                         _inp("Fed funds change, 12 months", r.fed_chg12, 2, " pts"),
                         _inp("Yield curve (10y minus 3m)", r.curve, 2, " pts"),
                         _inp("CFNAI 3-mo avg", r.cfnai_ma3)]),
    Framework("yield_curve", "Yield curve recession model", "recession", 1.0,
              "Recession probability 12 months out from the 10-year minus 3-month spread. Under 20%: +1. 40% or more: -1.",
              v_curve,
              lambda r, m: f"{curve_prob(m).iloc[-1]:.0f}% recession probability",
              lambda r: [_inp("10-year yield", r.y10, 2, "%"), _inp("3-month yield", r.y3m, 2, "%"),
                         _inp("Spread", r.curve, 2, " pts")]),
    Framework("sahm", "Sahm rule", "recession", 1.0,
              "Rise in 3-month average unemployment over its 12-month low. Under 0.30: +1. 0.50 or more: -1.",
              v_sahm,
              lambda r, m: f"Sahm reading {r.sahm:.2f}",
              lambda r: [_inp("Sahm rule", r.sahm), _inp("Unemployment rate", r.unrate, 1, "%")]),
    Framework("cfnai_ma3", "Leading activity (CFNAI)", "recession", 1.0,
              "CFNAI 3-month average. Zero or above: +1. Below -0.70: -1.",
              v_cfnai,
              lambda r, m: f"CFNAI 3-month average {r.cfnai_ma3:+.2f}",
              lambda r: [_inp("CFNAI 3-mo avg", r.cfnai_ma3), _inp("Industrial production, yearly change", r.indpro_yoy, 1, "%")]),
    Framework("claims", "Jobless claims trend", "recession", 0.75,
              "Initial claims (4-week average) versus a year ago. Up less than 10%: +1. Up more than 20%: -1.",
              v_claims,
              lambda r, m: f"Claims {r.claims_yoy:+.0f}% from a year ago",
              lambda r: [_inp("Initial claims, 4-week avg", r.claims_4wk, 0), _inp("Change from a year ago", r.claims_yoy, 1, "%")]),
    Framework("fed_direction", "Don't fight the Fed", "policy", 0.75,
              "Fed funds target versus 6 months ago. Lower: +1. Higher: -1.",
              v_fed,
              lambda r, m: ("Fed is hiking" if r.fed_chg6 >= 0.25 else "Fed is cutting" if r.fed_chg6 <= -0.25 else "Fed is on hold"),
              lambda r: [_inp("Fed funds target (upper)", r.fed_funds, 2, "%"), _inp("Change over 6 months", r.fed_chg6, 2, " pts")]),
    Framework("taylor_gap", "Taylor rule gap", "policy", 0.75,
              "Fed funds minus the Taylor rule rate. More than 1 pt above: -1. More than 1 pt below with inflation at or under 3%: +1.",
              v_taylor,
              lambda r, m: f"Policy is {abs(r.taylor_gap):.1f} pts {'tighter' if r.taylor_gap > 0 else 'looser'} than the rule",
              lambda r: [_inp("Fed funds", r.fed_funds, 2, "%"), _inp("Taylor rule rate", r.taylor_rate, 2, "%"),
                         _inp("Core PCE inflation", r.core_pce_yoy, 1, "%"), _inp("Unemployment gap", r.nrou - r.unrate, 1, " pts")]),
    Framework("fin_conditions", "Financial conditions", "policy", 0.75,
              "Chicago Fed NFCI. At or below zero and not tightening fast: +1. Above zero: -1.",
              v_nfci,
              lambda r, m: f"NFCI {r.nfci:+.2f} ({'looser' if r.nfci <= 0 else 'tighter'} than average)",
              lambda r: [_inp("NFCI", r.nfci), _inp("Change over 3 months", r.nfci_chg3)]),
    Framework("buffett", "Buffett indicator", "valuation", 0.75,
              "Market value of US stocks as a share of GDP, ranked against its own history. Below the 40th percentile: +1. Above the 80th: -1.",
              v_buffett,
              lambda r, m: f"Stocks are {r.buffett:.0f}% of GDP, {r.buffett_pctile:.0f}th percentile of history",
              lambda r: [_inp("Equity market value / GDP", r.buffett, 0, "%"), _inp("Percentile of own history", r.buffett_pctile, 0)]),
    Framework("trend_10m", "10-month trend rule", "trend", 1.0,
              "S&P 500 month-end close at or above its 10-month average: +1. Below: -1.",
              v_trend,
              lambda r, m: f"S&P 500 is {abs(r.sp500 / r.sp500_sma10 - 1) * 100:.1f}% {'above' if r.sp500 >= r.sp500_sma10 else 'below'} its 10-month average",
              lambda r: [_inp("S&P 500 month-end close", r.sp500, 0), _inp("10-month average", r.sp500_sma10, 0)]),
    Framework("golden_cross", "Golden cross / death cross", "trend", 0.5,
              "50-day average above the 200-day average: +1. Below: -1.",
              v_cross,
              lambda r, m: ("Golden cross: 50-day above 200-day" if r.sp500_sma50d >= r.sp500_sma200d else "Death cross: 50-day below 200-day"),
              lambda r: [_inp("50-day average", r.sp500_sma50d, 0), _inp("200-day average", r.sp500_sma200d, 0)]),
    Framework("abs_momentum", "Absolute momentum", "trend", 0.75,
              "S&P 500 12-month price return above the T-bill yield of a year ago: +1. Below: -1.",
              v_momentum,
              lambda r, m: f"Stocks {r.sp500_ret12:+.1f}% over 12 months vs {r.bill_12m_ago:.1f}% in T-bills",
              lambda r: [_inp("S&P 500 12-month return (price only)", r.sp500_ret12, 1, "%"), _inp("3-month T-bill yield a year ago", r.bill_12m_ago, 2, "%")]),
    Framework("hy_spread", "High-yield spread regime", "credit_vol", 1.0,
              "Spread under 4% and near or below its 12-month average: +1. Above both 5% and its 12-month average: -1.",
              v_credit,
              lambda r, m: (f"High-yield spread {r.hy_oas:.2f}%" if pd.notna(r.hy_oas) and pd.notna(r.hy_oas_sma12)
                            else f"Credit subindex {r.nfci_credit:+.2f} (spread history unavailable)"),
              lambda r: [_inp("High-yield spread", r.hy_oas, 2, "%"), _inp("12-month average", r.hy_oas_sma12, 2, "%"),
                         _inp("Chicago Fed credit subindex", r.nfci_credit)]),
    Framework("vix", "VIX level and term structure", "credit_vol", 0.75,
              "One-month average VIX under 20 with VIX below 3-month VIX: +1. Average above 30: -1.",
              v_vix,
              lambda r, m: f"VIX one-month average {r.vix_avg:.1f}",
              lambda r: [_inp("VIX, one-month average", r.vix_avg, 1), _inp("VIX, month-end", r.vix, 1),
                         _inp("VIX / 3-month VIX", r.vix_ratio)]),
    Framework("seasonality", "Seasonality and election cycle", "calendar", 0.5,
              "Next month in November to April adds 1. Post-midterm window adds 1; midterm year before November subtracts 1. Total of 1 or more: +1. Below zero: -1.",
              v_season,
              lambda r, m: _season_reading(m.index[-1]),
              lambda r: []),
]


def _season_reading(month_end: pd.Timestamp) -> str:
    best_six, cycle = seasonality_parts(pd.DatetimeIndex([month_end]))
    nxt = month_end + pd.offsets.MonthEnd(1)
    parts = ["inside the November to April window" if best_six.iloc[0] else "outside the November to April window"]
    parts.append({1.0: "post-midterm window", -1.0: "midterm year before November", 0.0: "neutral cycle year"}[cycle.iloc[0]])
    return f"{nxt.strftime('%B %Y')}: " + ", ".join(parts)


# ----------------------------------------------------------------------------
# Consensus
# ----------------------------------------------------------------------------
def compute_votes(m: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({f.id: f.vote(m) for f in FRAMEWORKS}, index=m.index)


def family_scores(votes: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for fam in FAMILIES:
        members = [f for f in FRAMEWORKS if f.family == fam]
        v = votes[[f.id for f in members]]
        w = pd.Series({f.id: f.weight for f in members})
        num = (v * w).sum(axis=1, min_count=1)
        den = (v.notna() * w).sum(axis=1)
        score = num / den.where(den > 0)
        if FAMILIES[fam][2] == "alarm":
            score = (score - 1.0) / 2.0   # clear = 0, watch = -0.5, triggered = -1
        out[fam] = score
    return pd.DataFrame(out, index=votes.index)


def consensus(votes: pd.DataFrame) -> pd.DataFrame:
    """Composite score, agreement, breakers and the confirmed stance for every month."""
    fam = family_scores(votes)
    drivers = [k for k, v in FAMILIES.items() if v[2] == "driver"]
    alarms = [k for k, v in FAMILIES.items() if v[2] == "alarm"]

    def wavg(cols):
        w = pd.Series({k: FAMILIES[k][1] for k in cols})
        f = fam[cols]
        return (f * w).sum(axis=1, min_count=1) / (f.notna() * w).sum(axis=1)

    driver_score = wavg(drivers).where(fam[drivers].notna().sum(axis=1) >= MIN_DRIVERS)
    alarm_penalty = wavg(alarms).fillna(0.0)
    composite = (driver_score + alarm_penalty).clip(-1.0, 1.0)

    ft = fam[drivers + alarms]
    sign = np.sign(composite)
    opinionated = ft.abs() >= 0.25
    agree = (np.sign(ft).eq(sign, axis=0) & opinionated).sum(axis=1)
    agreement = (agree / opinionated.sum(axis=1).where(opinionated.sum(axis=1) > 0)).fillna(0.0)

    raw = pd.Series(1, index=votes.index)  # index into STANCES, 1 = Neutral
    raw[(composite >= STANCE_THRESHOLD) & (agreement >= MIN_AGREEMENT)] = 2
    raw[(composite <= -STANCE_THRESHOLD) & (agreement >= MIN_AGREEMENT)] = 0

    breaker_recession = fam["recession"] <= BREAKER_LEVEL
    breaker_credit = votes["hy_spread"] == -1
    capped = raw.where(~((breaker_recession | breaker_credit) & (raw == 2)), 1)

    # Confirmation: a change needs two consecutive readings and moves one step at a time.
    # A breaker pulling Overweight back to Neutral applies at once.
    confirmed, current, prev = [], 1, None
    for t in votes.index:
        if pd.isna(composite[t]):
            confirmed.append(np.nan)
            prev = None
            continue
        target = int(capped[t])
        if (breaker_recession[t] or breaker_credit[t]) and current == 2:
            current = 1
        elif target != current and prev == target:
            current += 1 if target > current else -1
        prev = target
        confirmed.append(current)

    val = fam["valuation"]
    up = pd.Series(STEP, index=votes.index).where(~(val <= -0.5), VALUATION_STEP)
    down = pd.Series(STEP, index=votes.index).where(~(val >= 0.5), VALUATION_STEP)

    out = fam.add_prefix("fam_")
    out["driver_score"] = driver_score
    out["alarm_penalty"] = alarm_penalty
    out["composite"] = composite
    out["agreement"] = agreement
    out["raw_stance"] = raw.where(composite.notna())
    out["stance"] = confirmed
    out["breaker_recession"] = breaker_recession
    out["breaker_credit"] = breaker_credit
    out["step_up"] = up
    out["step_down"] = down
    stance = out["stance"]
    out["equity_offset"] = np.where(stance == 2, up, np.where(stance == 0, -down, 0.0))
    out.loc[stance.isna(), "equity_offset"] = np.nan
    fed = votes["fed_direction"]
    out["defensive"] = np.where(fed == -1, "cash", np.where(fed == 1, "bonds", "split"))
    return out


def stance_name(i) -> str | None:
    return None if i is None or pd.isna(i) else STANCES[int(i)]


# ----------------------------------------------------------------------------
# Asset-class playbooks and the model allocation
# ----------------------------------------------------------------------------
ASSETS = ["stocks", "bonds", "cash", "alternatives"]
NEUTRAL_MIX = {"stocks": 60, "bonds": 30, "cash": 5, "alternatives": 5}
ALT_STEP = 5                  # points added to or taken from alternatives
CASH_SHARE = {"cash": 0.5, "bonds": 0.1, "split": None}   # cash share of the bond-plus-cash sleeve

_QUADRANT_PLAY = {
    "Goldilocks": (["stocks"], "Growth up, inflation down: the classic backdrop for stocks and corporate credit."),
    "Reflation": (["stocks", "alternatives"], "Growth and inflation both up: stocks and commodities, light on long bonds."),
    "Stagflation": (["alternatives", "cash"], "Growth down, inflation up: commodities, gold and cash; hard on both stocks and bonds."),
    "Deflation": (["bonds", "cash"], "Growth and inflation both down: Treasuries and cash, defensive stocks only."),
}
_CLOCK_PLAY = {
    "Recovery": (["stocks"], "Above-trend growth with falling inflation: stocks lead."),
    "Overheat": (["alternatives"], "Above-trend growth with rising inflation: commodities lead, bonds lag."),
    "Stagflation": (["cash"], "Below-trend growth with rising inflation: cash is the refuge."),
    "Reflation": (["bonds"], "Below-trend growth with falling inflation: bonds lead as rates fall."),
}
_CYCLE_PLAY = {
    "Early": (["stocks"], "Early cycle: the strongest phase for stocks, especially economically sensitive ones."),
    "Mid": (["stocks"], "Mid cycle: stocks still lead, with smaller gains and more frequent corrections."),
    "Late": (["alternatives", "cash"], "Late cycle: commodities and short-term holdings, less stock and long-bond exposure."),
    "Recession": (["bonds", "cash"], "Recession: high-quality bonds and cash until growth turns."),
}


def asset_views(m: pd.DataFrame, votes: pd.DataFrame) -> list[dict]:
    """What each framework with an asset-class playbook favors in the latest month."""
    v = votes.iloc[-1]
    out = []

    def add(fid, favors, note):
        out.append({"id": fid, "favors": favors, "note": note})

    q, c, cy = quadrant_label(m).iloc[-1], clock_label(m).iloc[-1], cycle_label(m).iloc[-1]
    if isinstance(q, str) and q in _QUADRANT_PLAY:
        add("quadrant", *_QUADRANT_PLAY[q])
    if isinstance(c, str) and c in _CLOCK_PLAY:
        add("investment_clock", *_CLOCK_PLAY[c])
    if isinstance(cy, str) and cy in _CYCLE_PLAY:
        add("business_cycle", *_CYCLE_PLAY[cy])
    if not pd.isna(v["fed_direction"]):
        add("fed_direction", *{
            -1.0: (["cash"], "Fed hiking: short-term holdings earn more each meeting while longer bonds lose value."),
            1.0: (["bonds", "stocks"], "Fed cutting: bonds gain as yields fall and stocks get a tailwind."),
            0.0: (["bonds", "cash"], "Fed on hold: no strong case for cash over bonds or the reverse."),
        }[v["fed_direction"]])
    if not pd.isna(v["trend_10m"]):
        add("trend_10m", *{
            1.0: (["stocks"], "Stocks are in an uptrend: stay invested."),
            -1.0: (["cash", "bonds"], "Stocks are below trend: the rule moves that exposure to cash or bonds."),
        }[v["trend_10m"]])
    if not pd.isna(v["abs_momentum"]):
        add("abs_momentum", *{
            1.0: (["stocks"], "Stocks have beaten T-bills over 12 months: hold stocks."),
            -1.0: (["bonds"], "Stocks have lagged T-bills over 12 months: the rule holds bonds instead."),
        }[v["abs_momentum"]])
    rec = [v[k] for k in ("yield_curve", "sahm", "cfnai_ma3", "claims") if not pd.isna(v[k])]
    if rec and min(rec) == -1:
        add("sahm" if v["sahm"] == -1 else "yield_curve", ["bonds", "cash"],
            "A recession alarm is triggered: high-quality bonds and cash have held up best in recessions.")
    if v["hy_spread"] == -1:
        add("hy_spread", ["cash", "bonds"], "Credit stress: Treasuries and cash over corporate credit and stocks.")
    if not pd.isna(v["buffett"]) and v["buffett"] != 0:
        add("buffett", *{
            -1.0: (["bonds", "cash"], "Stocks are expensive against the economy: expect lower long-run stock returns and keep stock bets small."),
            1.0: (["stocks"], "Stocks are cheap against the economy: long-run stock returns should be above average."),
        }[v["buffett"]])
    return out


def alt_tilt(m: pd.DataFrame) -> float:
    """Alternatives (commodities, gold) get more room when inflation is accelerating
    and less when both growth and inflation are falling."""
    r = m.iloc[-1]
    if pd.isna(r["infl_chg"]):
        return 0.0
    if r["infl_chg"] > 0:
        return float(ALT_STEP)
    if not pd.isna(r["growth_chg"]) and r["growth_chg"] <= 0:
        return float(-ALT_STEP)
    return 0.0


def model_allocation(neutral: dict, equity_offset: float, alt_tilt_pts: float, defensive: str) -> dict:
    """Turn the consensus into a four-sleeve mix.

    1. Stocks move from neutral by the stance step.
    2. Alternatives move by the inflation tilt (never below zero).
    3. What is left is split between bonds and cash: half in cash while the Fed is
       hiking, about a tenth once it is cutting, the neutral split otherwise.
    """
    stocks = min(max(neutral["stocks"] + (equity_offset or 0), 0), 100)
    alts = min(max(neutral["alternatives"] + alt_tilt_pts, 0), 100 - stocks)
    rest = 100 - stocks - alts
    share = CASH_SHARE.get(defensive)
    if share is None:
        base = neutral["bonds"] + neutral["cash"]
        share = neutral["cash"] / base if base else 0.0
    cash = round(rest * share)
    return {"stocks": round(stocks), "bonds": round(rest - cash), "cash": cash, "alternatives": round(alts)}

