"""Backtest: what following each framework, and the consensus allocation, would have returned.

Read docs/backtest.html before trusting any number here. In short: the history is
short, the rules were written with that history already known, and results are before
taxes. This is a report card, not a tool for choosing parameters.

Mechanics
- Monthly. A vote or allocation decided at one month-end is held for the following month,
  so nothing uses information from the month it earns.
- Stocks, bonds, gold and commodities use ETF total returns. Cash earns the 3-month
  Treasury yield. "Alternatives" is half gold, half commodities.
- A trading cost of 0.10% is charged on the amount moved at each change.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

import frameworks as fw

HORIZONS = {"3m": 3, "6m": 6, "1y": 12, "3y": 36, "5y": 60, "10y": 120}
COST = 0.001                 # per dollar moved
MIN_MONTHS = 36              # shortest history worth reporting
MIN_WINDOWS = 12             # rolling windows needed before an average is shown
FOLLOW = {1.0: {"stocks": 1.0}, 0.0: {"stocks": 0.6, "bonds": 0.4}, -1.0: {"cash": 1.0}}


def asset_returns(m: pd.DataFrame) -> pd.DataFrame:
    """Monthly total returns for the four sleeves."""
    r = pd.DataFrame(index=m.index)
    r["stocks"] = m["px_SPY"].pct_change(fill_method=None)
    r["bonds"] = m["px_AGG"].pct_change(fill_method=None)
    r["cash"] = m["y3m"].shift(1) / 100.0 / 12.0
    gold = m["px_GLD"].pct_change(fill_method=None)
    comm = m["px_DBC"].pct_change(fill_method=None)
    r["alternatives"] = pd.concat([gold, comm], axis=1).mean(axis=1, skipna=True)
    return r


def run(weights: pd.DataFrame, rets: pd.DataFrame, cost: float = COST) -> pd.Series:
    """Returns of a strategy whose weights are decided at each month-end and held for the
    next month. Months where a held asset has no return, or no weights exist, are dropped."""
    weights = weights.reindex(columns=fw.ASSETS).fillna(0.0).where(weights.notna().any(axis=1))
    held = weights.shift(1)
    needed = (held > 0) & rets[fw.ASSETS].isna()
    gross = (held * rets[fw.ASSETS].fillna(0.0)).sum(axis=1, min_count=1)
    moved = (weights - weights.shift(1)).abs().sum(axis=1, min_count=1) / 2.0
    net = gross - moved.shift(1).fillna(0.0) * cost
    net[held.isna().any(axis=1) | needed.any(axis=1)] = np.nan
    return net.dropna()


def _longest_run(r: pd.Series) -> pd.Series:
    """The most recent unbroken stretch of monthly returns (rolling windows need contiguity)."""
    if r.empty:
        return r
    gaps = r.index.to_series().diff() > pd.Timedelta(days=35)
    start = r.index[gaps.values][-1] if gaps.any() else r.index[0]
    return r.loc[start:]


def stats(r: pd.Series, cash: pd.Series | None = None) -> dict | None:
    r = _longest_run(r.dropna())
    n = len(r)
    if n < MIN_MONTHS:
        return None
    growth = (1 + r).cumprod()
    out = {
        "start": r.index[0].strftime("%Y-%m"), "end": r.index[-1].strftime("%Y-%m"), "months": n,
        "cagr": growth.iloc[-1] ** (12 / n) - 1,
        "volatility": r.std() * math.sqrt(12),
        "max_drawdown": (growth / growth.cummax() - 1).min(),
        "worst_12m": None, "sharpe": None, "horizons": {},
    }
    if cash is not None and r.std() > 0:
        out["sharpe"] = (r - cash.reindex(r.index).fillna(0.0)).mean() / r.std() * math.sqrt(12)
    for name, h in HORIZONS.items():
        total = (1 + r).rolling(h).apply(np.prod, raw=True).dropna() - 1
        if len(total) < MIN_WINDOWS:
            out["horizons"][name] = None
            continue
        out["horizons"][name] = (((1 + total) ** (12 / h) - 1) if h > 12 else total).mean()
        if h == 12:
            out["worst_12m"] = total.min()
    return out


def follow_weights(vote: pd.Series) -> pd.DataFrame:
    """Follow one framework alone: all stocks when positive, 60/40 when neutral, T-bills when negative."""
    w = pd.DataFrame(np.nan, index=vote.index, columns=fw.ASSETS)
    for v, mix in FOLLOW.items():
        rows = vote == v
        w.loc[rows, :] = 0.0
        for asset, share in mix.items():
            w.loc[rows, asset] = share
    return w


def forward_by_vote(vote: pd.Series, stock_ret: pd.Series) -> dict:
    """Average stock return over the months after each kind of vote. The cleanest test of
    whether a vote carries information."""
    out = {}
    fwd = {}
    for name, h in (("3m", 3), ("6m", 6), ("1y", 12)):
        fwd[name] = (1 + stock_ret).rolling(h).apply(np.prod, raw=True).shift(-h) - 1
    for v in (1.0, 0.0, -1.0):
        rows = vote == v
        entry = {"months": int(rows.sum())}
        for name, series in fwd.items():
            vals = series[rows].dropna()
            entry[name] = vals.mean() if len(vals) >= MIN_WINDOWS else None
        out[str(int(v))] = entry
    return out


def consensus_weights(m: pd.DataFrame, cons: pd.DataFrame, neutral: dict | None = None) -> pd.DataFrame:
    neutral = fw.NEUTRAL_MIX if neutral is None else neutral
    tilt = fw.alt_tilt_series(m)
    rows = []
    for t in cons.index:
        off = cons.at[t, "equity_offset"]
        if pd.isna(off):
            rows.append({a: np.nan for a in fw.ASSETS})
            continue
        mix = fw.model_allocation(neutral, float(off), float(tilt[t]), cons.at[t, "defensive"])
        rows.append({a: mix[a] / 100.0 for a in fw.ASSETS})
    return pd.DataFrame(rows, index=cons.index)


def _static(index, mix: dict) -> pd.DataFrame:
    return pd.DataFrame({a: mix.get(a, 0) / 100.0 for a in fw.ASSETS}, index=index)


def build(m: pd.DataFrame, votes: pd.DataFrame, cons: pd.DataFrame) -> dict:
    rets = asset_returns(m)
    cash = rets["cash"]
    series = {
        "consensus": ("Consensus model allocation", run(consensus_weights(m, cons), rets)),
        "neutral": ("Neutral mix 60/30/5/5, held steady", run(_static(m.index, fw.NEUTRAL_MIX), rets)),
        "sixty_forty": ("Classic 60/40", run(_static(m.index, {"stocks": 60, "bonds": 40}), rets)),
        "stocks": ("Stocks only", run(_static(m.index, {"stocks": 100}), rets)),
    }
    # compare the four portfolios over the window the consensus model actually covers
    common = _longest_run(series["consensus"][1]).index
    portfolios = []
    for pid, (name, r) in series.items():
        s = stats(r.reindex(common).dropna() if len(common) else r, cash)
        if s:
            portfolios.append({"id": pid, "name": name, **s})
    curve = []
    if len(common):
        g = {pid: (1 + r.reindex(common).fillna(0.0)).cumprod() for pid, (_, r) in series.items()}
        curve = [{"date": d.strftime("%Y-%m"), **{pid: round(float(g[pid][d]), 4) for pid in g}} for d in common]

    frameworks = []
    for f in fw.FRAMEWORKS:
        v = votes[f.id]
        r = run(follow_weights(v), rets)
        s = stats(r, cash)
        if not s:
            frameworks.append({"id": f.id, "name": f.name, "family": f.family, "stats": None})
            continue
        window = _longest_run(r.dropna()).index
        bench = {pid: stats(series[pid][1].reindex(window).dropna(), cash) for pid in ("stocks", "sixty_forty")}
        held = v.shift(1).reindex(window)
        frameworks.append({
            "id": f.id, "name": f.name, "family": f.family, "stats": s,
            "benchmarks": {k: ({"cagr": b["cagr"], "max_drawdown": b["max_drawdown"], "horizons": b["horizons"]} if b else None)
                           for k, b in bench.items()},
            "time": {"stocks": float((held == 1).mean()), "mixed": float((held == 0).mean()), "cash": float((held == -1).mean())},
            "switches_per_year": float((held != held.shift(1)).sum() - 1) / (len(window) / 12.0),
            "forward": forward_by_vote(v, rets["stocks"]),
        })
    return {
        "as_of": m.index[-1].strftime("%Y-%m-%d"),
        "cost_bps": COST * 10000,
        "horizons": list(HORIZONS),
        "follow_rule": "All stocks when the framework is positive (or its alarm is clear), 60/40 when neutral (watch), T-bills when negative (triggered).",
        "portfolios": portfolios,
        "curve": curve,
        "frameworks": frameworks,
    }
