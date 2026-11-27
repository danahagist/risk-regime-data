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
#   info    shows which parts of the market are leading; not part of the score
FAMILIES = {
    # id: (name, weight, role)
    "macro": ("Macro regime", 0.75, "driver"),
    "policy": ("Policy and liquidity", 0.75, "driver"),
    "trend": ("Trend and momentum", 1.0, "driver"),
    "calendar": ("Sentiment and calendar", 0.5, "driver"),
    "recession": ("Recession risk", 1.0, "alarm"),
    "credit_vol": ("Credit and volatility", 1.0, "alarm"),
    "valuation": ("Valuation", 0.75, "sizing"),
    "where": ("Where to look", 0.0, "info"),   # shown for context; never moves the stance
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


SECTORS = {
    "XLK": "Technology", "XLC": "Communication", "XLY": "Consumer discretionary", "XLF": "Financials",
    "XLI": "Industrials", "XLB": "Materials", "XLE": "Energy", "XLV": "Health care",
    "XLP": "Consumer staples", "XLU": "Utilities", "XLRE": "Real estate",
}
CYCLICAL = {"XLK", "XLC", "XLY", "XLF", "XLI", "XLB", "XLE"}
TREND_ASSETS = {"SPY": "US stocks", "ACWX": "Foreign stocks", "AGG": "US bonds", "GLD": "Gold", "DBC": "Commodities"}
PRICE_TICKERS = list(TREND_ASSETS) + ["BIL", "IWM", "RSP", "DIA", "IYT", "CPER"] + list(SECTORS)
RATIOS = {   # name: (numerator, denominator). Rising means markets expect stronger growth.
    "copper_gold": ("CPER", "GLD"),
    "discretionary_staples": ("XLY", "XLP"),
    "small_large": ("IWM", "SPY"),
}
RATIO_NAME = {"copper_gold": "Copper / gold", "discretionary_staples": "Discretionary / staples", "small_large": "Small caps / large caps"}


def _to_billions(s: pd.Series) -> pd.Series:
    """FRED mixes units across these series (millions for some, billions for others)."""
    med = s.abs().median()
    return s / 1000.0 if pd.notna(med) and med > 50_000 else s


def splice_sp500(raw: dict) -> pd.Series:
    """FRED carries about ten years of the S&P 500. For earlier history, use an S&P 500
    fund's price scaled to meet the index where FRED's series begins. The fund's history
    includes dividends, which is fine for trend and momentum rules."""
    spx = raw.get("SP500", pd.Series(dtype="float64")).dropna().sort_index()
    spy = raw.get("PX:SPY", pd.Series(dtype="float64")).dropna().sort_index()
    if spx.empty or spy.empty or spy.index[0] >= spx.index[0]:
        return spx
    at = spy.asof(spx.index[0])
    if pd.isna(at) or at <= 0:
        return spx
    return pd.concat([spy[spy.index < spx.index[0]] * (spx.iloc[0] / at), spx])


def _month_end(raw, key) -> pd.Series:
    s = raw.get(key, pd.Series(dtype="float64")).dropna().sort_index()
    if s.empty:
        return s
    return s.groupby(s.index.to_period("M")).last().to_timestamp("M")


def valuation_series(raw: dict, idx: pd.DatetimeIndex, sp500: pd.Series) -> pd.DataFrame:
    """CAPE, its percentile, and trailing P/E for each month in idx.

    If the published series are current, they are used as they are. Shiller's own file
    stopped updating in 2023, so otherwise recent months are estimated: earnings are
    interpolated between Shiller's last figure and the trailing earnings implied by
    today's fund P/E, CAPE and P/E are recomputed from those, and the estimates are
    chain-linked to Shiller's last published values so there is no jump at the join.
    """
    out = pd.DataFrame({"cape": np.nan, "cape_pctile": np.nan, "pe": np.nan, "valuation_estimated": False}, index=idx)
    cape, pe = _month_end(raw, "SHILLER:CAPE"), _month_end(raw, "SHILLER:PE")
    if cape.empty:
        return out
    asof = idx[-1]
    estimated = pd.Series(False, index=idx)
    if cape.index[-1] < asof - pd.offsets.MonthEnd(3):
        eps_hist, cpi_hist, px_hist = (_month_end(raw, k) for k in ("SHILLER:EPS", "SHILLER:CPI", "SHILLER:PRICE"))
        points = _month_end(raw, "VAL:EPS_TTM")
        if not (points.empty or eps_hist.empty):
            points = points[points.index > eps_hist.index[-1]]
        if not (eps_hist.empty or cpi_hist.empty or px_hist.empty or points.empty):
            # Trailing earnings move slowly, and a run on the 2nd observes them a few days
            # after the month it reports on, so let the latest point anchor the interpolation
            # even when it is dated just past the as-of month.
            span = pd.date_range(eps_hist.index[0], max(asof, points.index[-1]), freq="ME")
            eps = pd.concat([eps_hist, points]).reindex(span).interpolate(limit_area="inside").ffill(limit=2)
            full = pd.date_range(eps_hist.index[0], asof, freq="ME")
            eps = eps.reindex(full)
            price = sp500.reindex(full).combine_first(px_hist.reindex(full))
            cpi_new = _month_end(raw, "CPIAUCSL").reindex(full).ffill()
            link = cpi_hist.index[-1]
            if pd.notna(cpi_new.get(link, np.nan)):
                cpi = cpi_hist.reindex(full).combine_first(cpi_new * (cpi_hist.iloc[-1] / cpi_new[link]))
            else:
                cpi = cpi_hist.reindex(full).ffill()
            cpi = cpi.ffill()
            real_e = eps / cpi
            cape_est = (price / cpi) / real_e.rolling(120, min_periods=120).mean()
            pe_est = price / eps

            def chain(published, est):
                at = published.index[-1]
                if pd.isna(est.get(at, np.nan)) or est[at] == 0:
                    return published
                tail = est[est.index > at] * (published.iloc[-1] / est[at])
                return pd.concat([published, tail.dropna()])

            n_before = len(cape)
            cape, pe = chain(cape, cape_est), (chain(pe, pe_est) if not pe.empty else pe_est.dropna())
            if len(cape) > n_before:
                estimated = pd.Series(idx > _month_end(raw, "SHILLER:CAPE").index[-1], index=idx)
    pct = cape.expanding(min_periods=120).apply(lambda w: (w <= w.iloc[-1]).mean() * 100, raw=False)
    out["cape"] = cape.reindex(idx).ffill(limit=2)
    out["cape_pctile"] = pct.reindex(idx).ffill(limit=2)
    if not pe.empty:
        out["pe"] = pe.reindex(idx).ffill(limit=2)
    out["valuation_estimated"] = estimated & out["cape"].notna()
    return out


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
    spx_d = splice_sp500(raw)
    m["sp500"] = _daily({"SP500": spx_d}, "SP500", idx, "last")
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

    # ETF prices (adjusted closes, so returns include dividends)
    cols = {}
    for t in PRICE_TICKERS:
        px = _daily(raw, f"PX:{t}", idx, "last")
        cols[f"px_{t}"] = px
        cols[f"sma_{t}"] = px.rolling(10, min_periods=10).mean()
        cols[f"ret6_{t}"] = px.pct_change(6, fill_method=None) * 100
        cols[f"ret12_{t}"] = px.pct_change(12, fill_method=None) * 100
    m = pd.concat([m, pd.DataFrame(cols, index=idx)], axis=1)
    sector_up = pd.DataFrame({t: (m[f"px_{t}"] >= m[f"sma_{t}"]).where(_ok(m[f"px_{t}"], m[f"sma_{t}"])) for t in SECTORS})
    m["sectors_counted"] = sector_up.notna().sum(axis=1)
    m["breadth"] = (sector_up.astype("float64").mean(axis=1) * 100).where(m["sectors_counted"] >= 8)
    for name, (a, b) in RATIOS.items():
        r = m[f"px_{a}"] / m[f"px_{b}"]
        m[f"ratio_{name}"] = r
        m[f"ratio_{name}_sma"] = r.rolling(10, min_periods=10).mean()

    # long-run valuation
    val = valuation_series(raw, idx, m["sp500"])
    for k in ("cape", "cape_pctile", "pe", "valuation_estimated"):
        m[k] = val[k]
    m["real_y10"] = _daily(raw, "DFII10", idx, "mean")
    m["excess_cape_yield"] = 100.0 / m["cape"] - m["real_y10"]
    m["earnings_yield"] = 100.0 / m["pe"]
    m["fed_model_gap"] = m["earnings_yield"] - m["y10"]
    m["rule20"] = m["pe"] + m["cpi_yoy"]

    # sentiment: AAII bulls minus bears, 4-week average
    aaii = raw.get("AAII:SPREAD", pd.Series(dtype="float64")).dropna().sort_index()
    if aaii.empty:
        m["aaii_spread"] = np.nan
    else:
        m["aaii_spread"] = aaii.rolling(4, min_periods=2).mean().resample("ME").last().reindex(idx)

    # liquidity: Fed balance sheet minus the Treasury's cash account minus reverse repo
    parts = [_to_billions(_daily(raw, c, idx, "last")) for c in ("WALCL", "WTREGEN", "RRPONTSYD")]
    m["net_liquidity"] = parts[0] - parts[1] - parts[2]
    m["net_liquidity_chg6"] = m["net_liquidity"].pct_change(6, fill_method=None) * 100
    return m.copy()


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


# ---- price-based frameworks -------------------------------------------------
def dual_momentum_holding(m):
    """Antonacci's rule: if US stocks beat T-bills over 12 months, hold US or foreign
    stocks, whichever returned more. Otherwise hold bonds."""
    us, intl, bills = m["ret12_SPY"], m["ret12_ACWX"], m["ret12_BIL"]
    lab = np.select([(us > bills) & (us >= intl), (us > bills) & (intl > us)], ["US stocks", "Foreign stocks"], default="Bonds")
    return pd.Series(lab, index=m.index).where(_ok(us, intl, bills))


def v_dual_momentum(m):
    h = dual_momentum_holding(m)
    return _tri(h.isin(["US stocks", "Foreign stocks"]), h == "Bonds", h.notna())


def v_breadth(m):
    return _tri(m["breadth"] >= 60, m["breadth"] <= 40, m["breadth"].notna())


def v_dow(m):
    ind = m["px_DIA"] >= m["sma_DIA"]
    tra = m["px_IYT"] >= m["sma_IYT"]
    return _tri(ind & tra, ~ind & ~tra, _ok(m["px_DIA"], m["sma_DIA"], m["px_IYT"], m["sma_IYT"]))


def assets_in_uptrend(m):
    up = pd.DataFrame({t: (m[f"px_{t}"] >= m[f"sma_{t}"]).where(_ok(m[f"px_{t}"], m[f"sma_{t}"])) for t in TREND_ASSETS})
    return up


def v_gtaa(m):
    up = assets_in_uptrend(m)
    n = up.astype("float64").sum(axis=1)
    return _tri(n >= 3, n <= 1, up.notna().all(axis=1))


def sector_scores(m) -> pd.DataFrame:
    """Average of 6 and 12-month total return for each sector."""
    return pd.DataFrame({t: (m[f"ret6_{t}"] + m[f"ret12_{t}"]) / 2 for t in SECTORS})


def v_sectors(m):
    s = sector_scores(m)
    valid = s.notna().sum(axis=1) >= 8
    top3_cyc = s.rank(axis=1, ascending=False).le(3)[[t for t in SECTORS if t in CYCLICAL]].sum(axis=1)
    return _tri(top3_cyc >= 2, top3_cyc <= 1, valid)


def ratios_rising(m):
    return pd.DataFrame({k: (m[f"ratio_{k}"] >= m[f"ratio_{k}_sma"]).where(_ok(m[f"ratio_{k}"], m[f"ratio_{k}_sma"])) for k in RATIOS})


def v_intermarket(m):
    up = ratios_rising(m)
    n = up.astype("float64").sum(axis=1)
    return _tri(n >= 2, n == 0, up.notna().all(axis=1))


# ---- valuation, sentiment, liquidity ----------------------------------------
def v_cape(m):
    return _tri(m["cape_pctile"] < 40, m["cape_pctile"] > 80, m["cape_pctile"].notna())


def v_fed_model(m):
    return _tri(m["fed_model_gap"] > 2.0, m["fed_model_gap"] < 0.0, m["fed_model_gap"].notna())


def v_rule20(m):
    return _tri(m["rule20"] < 18, m["rule20"] > 22, m["rule20"].notna())


def v_sentiment(m):
    # contrarian: deep pessimism is a positive, euphoria a negative
    return _tri(m["aaii_spread"] <= -20, m["aaii_spread"] >= 30, m["aaii_spread"].notna())


def v_liquidity(m):
    return _tri(m["net_liquidity_chg6"] > 2.0, m["net_liquidity_chg6"] < -2.0, m["net_liquidity_chg6"].notna())


# ---- tables shown with a framework's reading --------------------------------
def _updown(flag):
    return "n/a" if pd.isna(flag) else ("Above" if flag else "Below")


def t_gtaa(r, m):
    return {"columns": ["Asset class", "vs 10-month average", "12-month return"],
            "rows": [[TREND_ASSETS[t], _updown(None if pd.isna(r[f"px_{t}"]) or pd.isna(r[f"sma_{t}"]) else r[f"px_{t}"] >= r[f"sma_{t}"]),
                      _f(r[f"ret12_{t}"], 1, "%") or "n/a"] for t in TREND_ASSETS]}


def t_sectors(r, m):
    s = sector_scores(m).iloc[-1].dropna().sort_values(ascending=False)
    return {"columns": ["Rank", "Sector", "Type", "6 and 12-month return, averaged", "vs 10-month average"],
            "rows": [[i + 1, SECTORS[t], "Cyclical" if t in CYCLICAL else "Defensive", _f(v, 1, "%"),
                      _updown(None if pd.isna(r[f"px_{t}"]) or pd.isna(r[f"sma_{t}"]) else r[f"px_{t}"] >= r[f"sma_{t}"])]
                     for i, (t, v) in enumerate(s.items())]}


def t_intermarket(r, m):
    return {"columns": ["Ratio", "vs its 10-month average", "Says"],
            "rows": [[RATIO_NAME[k], _updown(None if pd.isna(r[f"ratio_{k}"]) or pd.isna(r[f"ratio_{k}_sma"]) else r[f"ratio_{k}"] >= r[f"ratio_{k}_sma"]),
                      "n/a" if pd.isna(r[f"ratio_{k}_sma"]) else ("Growth improving" if r[f"ratio_{k}"] >= r[f"ratio_{k}_sma"] else "Growth weakening")]
                     for k in RATIOS]}


def _sector_reading(m):
    s = sector_scores(m).iloc[-1].dropna().sort_values(ascending=False)
    names = [SECTORS[t] for t in s.index]
    return f"Leading: {', '.join(names[:3])}. Lagging: {', '.join(names[-3:])}"


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
    table: Callable[[pd.Series, pd.DataFrame], dict] | None = None


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
    # ---- price data
    Framework("dual_momentum", "Dual momentum", "trend", 0.75,
              "If US stocks beat T-bills over 12 months, hold US or foreign stocks, whichever is stronger: +1. Otherwise hold bonds: -1.",
              v_dual_momentum,
              lambda r, m: f"Holds {dual_momentum_holding(m).iloc[-1]}",
              lambda r: [_inp("US stocks, 12-month return", r.ret12_SPY, 1, "%"), _inp("Foreign stocks, 12-month return", r.ret12_ACWX, 1, "%"),
                         _inp("T-bills, 12-month return", r.ret12_BIL, 1, "%")]),
    Framework("breadth", "Market breadth", "trend", 0.75,
              "Share of the 11 sectors above their 10-month average. 60% or more: +1. 40% or less: -1.",
              v_breadth,
              lambda r, m: f"{r.breadth:.0f}% of sectors are above their 10-month average",
              lambda r: [_inp("Sectors above their 10-month average", r.breadth, 0, "%"), _inp("Sectors counted", r.sectors_counted, 0)]),
    Framework("dow_theory", "Dow Theory", "trend", 0.5,
              "Dow industrials and transports both above their 10-month averages: +1. Both below: -1. Split: 0.",
              v_dow,
              lambda r, m: {(True, True): "Industrials and transports both in uptrends", (False, False): "Industrials and transports both in downtrends"}.get(
                  (bool(r.px_DIA >= r.sma_DIA), bool(r.px_IYT >= r.sma_IYT)), "Industrials and transports disagree"),
              lambda r: [_inp("Industrials vs 10-month average", (r.px_DIA / r.sma_DIA - 1) * 100, 1, "%"),
                         _inp("Transports vs 10-month average", (r.px_IYT / r.sma_IYT - 1) * 100, 1, "%")]),
    # ---- valuation
    Framework("cape", "Shiller CAPE and excess CAPE yield", "valuation", 0.75,
              "CAPE ranked against its own history. Below the 40th percentile: +1. Above the 80th: -1.",
              v_cape,
              lambda r, m: f"CAPE {'about ' if r.valuation_estimated else ''}{r.cape:.1f}{' (estimated)' if r.valuation_estimated else ''}, {r.cape_pctile:.0f}th percentile of history",
              lambda r: [_inp("CAPE", r.cape, 1), _inp("Percentile of own history", r.cape_pctile, 0),
                         _inp("Real 10-year Treasury yield", r.real_y10, 2, "%"), _inp("Excess CAPE yield", r.excess_cape_yield, 2, "%")]),
    Framework("fed_model", "Fed model (equity risk premium)", "valuation", 0.75,
              "Trailing earnings yield minus the 10-year Treasury yield. Above 2 points: +1. Below zero: -1.",
              v_fed_model,
              lambda r, m: f"Stocks yield {abs(r.fed_model_gap):.1f} pts {'more' if r.fed_model_gap >= 0 else 'less'} than the 10-year Treasury",
              lambda r: [_inp("Trailing P/E", r.pe, 1), _inp("Earnings yield", r.earnings_yield, 2, "%"),
                         _inp("10-year Treasury yield", r.y10, 2, "%"), _inp("Gap", r.fed_model_gap, 2, " pts")]),
    Framework("rule_of_20", "Rule of 20", "valuation", 0.5,
              "Trailing P/E plus CPI inflation. Below 18: +1. Above 22: -1.",
              v_rule20,
              lambda r, m: f"P/E plus inflation is {r.rule20:.1f}",
              lambda r: [_inp("Trailing P/E", r.pe, 1), _inp("CPI inflation", r.cpi_yoy, 1, "%"), _inp("Sum", r.rule20, 1)]),
    # ---- sentiment and liquidity
    Framework("sentiment", "Contrarian sentiment", "calendar", 0.5,
              "AAII bulls minus bears, 4-week average. At -20 or lower: +1. At +30 or higher: -1.",
              v_sentiment,
              lambda r, m: f"Bulls minus bears: {r.aaii_spread:+.0f} points",
              lambda r: [_inp("AAII bull-bear spread, 4-week average", r.aaii_spread, 1, " pts")]),
    Framework("net_liquidity", "Liquidity", "policy", 0.75,
              "Fed balance sheet minus Treasury cash minus reverse repo, change over 6 months. Up more than 2%: +1. Down more than 2%: -1.",
              v_liquidity,
              lambda r, m: f"Net liquidity {r.net_liquidity_chg6:+.1f}% over 6 months",
              lambda r: [_inp("Net liquidity ($ billions)", r.net_liquidity, 0), _inp("Change over 6 months", r.net_liquidity_chg6, 1, "%")]),
    # ---- where to look (not part of the score)
    Framework("gtaa", "Multi-asset trend", "where", 0.75,
              "Each asset class against its own 10-month average. Three or more of five in uptrends: +1. One or none: -1.",
              v_gtaa,
              lambda r, m: "In uptrends: " + (", ".join(TREND_ASSETS[t] for t in TREND_ASSETS if r[f"px_{t}"] >= r[f"sma_{t}"]) or "none"),
              lambda r: [], t_gtaa),
    Framework("sector_rotation", "Sector momentum rotation", "where", 0.75,
              "Sectors ranked by their average 6 and 12-month return. Two or more cyclical sectors in the top three: +1. One or none: -1.",
              v_sectors,
              lambda r, m: _sector_reading(m),
              lambda r: [], t_sectors),
    Framework("intermarket", "Intermarket ratios", "where", 0.5,
              "Copper/gold, discretionary/staples and small/large against their 10-month averages. Two or more rising: +1. None rising: -1.",
              v_intermarket,
              lambda r, m: f"{int(ratios_rising(m).iloc[-1].astype('float64').sum())} of 3 growth ratios are rising",
              lambda r: [], t_intermarket),
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
    dm = dual_momentum_holding(m).iloc[-1]
    if isinstance(dm, str):
        add("dual_momentum", ["stocks"] if dm != "Bonds" else ["bonds"],
            f"Dual momentum currently holds {'US stocks' if dm == 'US stocks' else dm.lower()}.")
    up = assets_in_uptrend(m).iloc[-1]
    if up.notna().all():
        favors = []
        if up["SPY"] or up["ACWX"]:
            favors.append("stocks")
        if up["AGG"]:
            favors.append("bonds")
        if up["GLD"] or up["DBC"]:
            favors.append("alternatives")
        if int(up.astype("float64").sum()) <= 2:
            favors.append("cash")
        names = [TREND_ASSETS[t] for t in TREND_ASSETS if up[t]]
        add("gtaa", favors, ("In uptrends: " + ", ".join(names) + ".") if names else "No asset class is in an uptrend: the rule holds cash.")
    if not pd.isna(v["fed_model"]) and v["fed_model"] != 0:
        add("fed_model", *{
            -1.0: (["bonds"], "The 10-year Treasury yields more than stocks earn: bonds are the better deal on this measure."),
            1.0: (["stocks"], "Stocks earn well above the Treasury yield: the premium for owning them is wide."),
        }[v["fed_model"]])
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


def alt_tilt_series(m: pd.DataFrame) -> pd.Series:
    """The alternatives tilt for every month, in points. Same rule as alt_tilt()."""
    infl_up = m["infl_chg"] > 0
    known = _ok(m["px_GLD"], m["sma_GLD"], m["px_DBC"], m["sma_DBC"])
    trending = (m["px_GLD"] >= m["sma_GLD"]) | (m["px_DBC"] >= m["sma_DBC"])
    out = pd.Series(0.0, index=m.index)
    out[infl_up & (~known | trending)] = float(ALT_STEP)
    out[m["infl_chg"].notna() & ~infl_up & (m["growth_chg"] <= 0)] = float(-ALT_STEP)
    return out


def alt_tilt(m: pd.DataFrame) -> tuple[float, str]:
    """Alternatives (commodities, gold) get more room when inflation is accelerating and
    at least one of them is in an uptrend, and less when growth and inflation are both
    falling. Returns the tilt in points and a one-line reason."""
    r = m.iloc[-1]
    if pd.isna(r["infl_chg"]):
        return 0.0, "there is no inflation reading this month."
    if r["infl_chg"] > 0:
        trend_known = _ok(m["px_GLD"], m["sma_GLD"], m["px_DBC"], m["sma_DBC"]).iloc[-1]
        if trend_known and not (r["px_GLD"] >= r["sma_GLD"] or r["px_DBC"] >= r["sma_DBC"]):
            return 0.0, "inflation is accelerating, but neither gold nor commodities is in an uptrend."
        return float(ALT_STEP), "inflation is accelerating."
    if not pd.isna(r["growth_chg"]) and r["growth_chg"] <= 0:
        return float(-ALT_STEP), "growth and inflation are both slowing."
    return 0.0, "inflation is not accelerating."


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

