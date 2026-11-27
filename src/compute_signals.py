"""Compute framework votes and the consensus, and write docs/data/signals.json.

Usage:
    python src/compute_signals.py                    # live, needs FRED_API_KEY
    python src/compute_signals.py --fixture DIR      # offline, from CSV files
    python src/compute_signals.py --asof 2026-10-31  # evaluate an earlier month
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fred_data  # noqa: E402
import frameworks as fw  # noqa: E402

OUT_JSON = "docs/data/signals.json"
OUT_HISTORY = "data/consensus_history.csv"
VOTE_WORD = {1.0: "Positive", 0.0: "Neutral", -1.0: "Negative"}
ALARM_WORD = {1.0: "Clear", 0.0: "Watch", -1.0: "Triggered"}


def vote_word(framework, v) -> str:
    if pd.isna(v):
        return "No data"
    return (ALARM_WORD if fw.FAMILIES[framework.family][2] == "alarm" else VOTE_WORD)[v]

# Dashboard tiles: (key, group, label, digits, unit). Explanations live in docs/data/lessons.json.
METRICS = [
    ("cfnai_ma3", "Growth", "CFNAI, 3-month average", 2, ""),
    ("indpro_yoy", "Growth", "Industrial production, yearly change", 1, "%"),
    ("unrate", "Labor", "Unemployment rate", 1, "%"),
    ("claims_4wk", "Labor", "Initial claims, 4-week average", 0, ""),
    ("sahm", "Labor", "Sahm rule", 2, ""),
    ("core_pce_yoy", "Inflation", "Core PCE inflation", 1, "%"),
    ("cpi_yoy", "Inflation", "CPI inflation", 1, "%"),
    ("fed_funds", "Policy and rates", "Fed funds target (upper)", 2, "%"),
    ("real_ff", "Policy and rates", "Real policy rate", 2, "%"),
    ("taylor_gap", "Policy and rates", "Taylor rule gap", 2, " pts"),
    ("y10", "Policy and rates", "10-year Treasury yield", 2, "%"),
    ("y2", "Policy and rates", "2-year Treasury yield", 2, "%"),
    ("curve", "Policy and rates", "Yield curve, 10y minus 3m", 2, " pts"),
    ("curve_prob", "Policy and rates", "Recession probability (yield curve)", 0, "%"),
    ("nfci", "Policy and rates", "Financial conditions (NFCI)", 2, ""),
    ("hy_oas", "Credit and volatility", "High-yield spread", 2, "%"),
    ("vix", "Credit and volatility", "VIX", 1, ""),
    ("vix_ratio", "Credit and volatility", "VIX / 3-month VIX", 2, ""),
    ("sp500", "Markets", "S&P 500", 0, ""),
    ("sp500_vs_sma", "Markets", "S&P 500 vs 10-month average", 1, "%"),
    ("sp500_ret12", "Markets", "S&P 500, 12-month return", 1, "%"),
    ("buffett", "Valuation", "Stock market value / GDP", 0, "%"),
]


def _clean(x):
    """JSON-safe: NaN and NaT become null, numpy scalars become Python numbers."""
    if x is None:
        return None
    if isinstance(x, (bool, str)):
        return x
    if hasattr(x, "item"):
        x = x.item()
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else round(x, 4)
    return x


def build_payload(raw: dict, asof=None) -> tuple[dict, pd.DataFrame]:
    m = fw.build_monthly(raw, asof=asof)
    m["curve_prob"] = fw.curve_prob(m)
    m["real_ff"] = m["fed_funds"] - m["core_pce_yoy"]
    m["sp500_vs_sma"] = (m["sp500"] / m["sp500_sma10"] - 1) * 100
    votes = fw.compute_votes(m)
    cons = fw.consensus(votes)
    last, row, c = m.index[-1], m.iloc[-1], cons.iloc[-1]
    prev_votes = votes.iloc[-2] if len(votes) > 1 else None

    frameworks = []
    for f in fw.FRAMEWORKS:
        v = votes[f.id].iloc[-1]
        has = not pd.isna(v)
        try:
            reading = f.reading(row, m) if has else "No data this month"
        except Exception:  # a display string must never break the run
            reading = "Reading unavailable"
        frameworks.append({
            "id": f.id, "name": f.name, "family": f.family, "weight": f.weight,
            "vote": _clean(v), "vote_label": vote_word(f, v),
            "reading": reading, "rule": f.rule,
            "inputs": [i for i in f.inputs(row) if i["value"] is not None] if has else [],
        })

    families = []
    for fid, (name, weight, role) in fw.FAMILIES.items():
        families.append({
            "id": fid, "name": name, "weight": weight, "role": role,
            "score": _clean(c[f"fam_{fid}"]),
            "members": [f.id for f in fw.FRAMEWORKS if f.family == fid],
        })

    changes = []
    if prev_votes is not None:
        for f in fw.FRAMEWORKS:
            a, b = prev_votes[f.id], votes[f.id].iloc[-1]
            if not pd.isna(a) and not pd.isna(b) and a != b:
                changes.append(f"{f.name} moved from {vote_word(f, a)} to {vote_word(f, b)}")
        if len(cons) > 1:
            a, b = cons["stance"].iloc[-2], c["stance"]
            if not pd.isna(a) and not pd.isna(b) and a != b:
                changes.insert(0, f"Stance moved from {fw.stance_name(a)} to {fw.stance_name(b)}")

    breakers = []
    if bool(c["breaker_recession"]):
        breakers.append("Recession alarms are triggered: equity capped at Neutral")
    if bool(c["breaker_credit"]):
        breakers.append("Credit stress signal is on: equity capped at Neutral")

    val = c["fam_valuation"]
    tilt = "unknown" if pd.isna(val) else "expensive" if val <= -0.5 else "cheap" if val >= 0.5 else "neutral"

    metrics = []
    for key, group, label, digits, unit in METRICS:
        x = row.get(key)
        if x is None or pd.isna(x):
            continue
        avg12 = m[key].iloc[-13:-1].mean()
        metrics.append({
            "key": key, "group": group, "label": label, "value": fw._f(x, digits, unit),
            "avg12": fw._f(avg12, digits, unit),
            "vs_avg": None if pd.isna(avg12) else ("above" if x > avg12 else "below" if x < avg12 else "at"),
        })

    views = fw.asset_views(m, votes)
    weights = {f.id: f.weight for f in fw.FRAMEWORKS}
    tally = {a: round(sum(weights[v["id"]] for v in views if a in v["favors"]), 2) for a in fw.ASSETS}
    tilt_pts = fw.alt_tilt(m)
    offset = 0.0 if pd.isna(c["equity_offset"]) else float(c["equity_offset"])
    allocation = {
        "neutral": fw.NEUTRAL_MIX,
        "model": fw.model_allocation(fw.NEUTRAL_MIX, offset, tilt_pts, c["defensive"]),
        "equity_offset": offset, "alt_tilt": tilt_pts, "defensive": c["defensive"],
        "cash_share": fw.CASH_SHARE, "alt_step": fw.ALT_STEP,
        "views": views, "tally": tally,
    }

    hist = cons.dropna(subset=["composite"]).tail(120)
    history = [{
        "date": d.strftime("%Y-%m-%d"), "composite": _clean(r["composite"]),
        "stance": fw.stance_name(r["stance"]), "sp500": _clean(m.loc[d, "sp500"]),
    } for d, r in hist.iterrows()]

    payload = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "as_of": last.strftime("%Y-%m-%d"),
        "as_of_label": last.strftime("%B %Y"),
        "stance": {
            "label": fw.stance_name(c["stance"]) or "Not enough data",
            "raw_label": fw.stance_name(c["raw_stance"]),
            "composite": _clean(c["composite"]),
            "driver_score": _clean(c["driver_score"]),
            "alarm_penalty": _clean(c["alarm_penalty"]),
            "agreement": _clean(c["agreement"]),
            "equity_offset": _clean(c["equity_offset"]),
            "step_up": _clean(c["step_up"]), "step_down": _clean(c["step_down"]),
            "valuation_tilt": tilt,
            "defensive": c["defensive"],
            "breakers": breakers,
            "threshold": fw.STANCE_THRESHOLD,
        },
        "families": families,
        "frameworks": frameworks,
        "changes": changes,
        "allocation": allocation,
        "metrics": metrics,
        "history": history,
        "sources": [{"id": k, "name": v[0], "rights": v[1], "available": not raw.get(k, pd.Series(dtype=float)).empty}
                    for k, v in fred_data.SERIES.items()],
    }
    full = pd.concat([votes.add_prefix("vote_"), cons], axis=1)
    full["stance_label"] = full["stance"].map(fw.stance_name)
    return payload, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", help="directory of <CODE>.csv files to use instead of FRED")
    ap.add_argument("--asof", help="evaluate as of this month (YYYY-MM-DD)")
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--history-out", default=OUT_HISTORY)
    args = ap.parse_args()

    raw = fred_data.load_fixture(args.fixture) if args.fixture else fred_data.fetch_all()
    payload, full = build_payload(raw, asof=args.asof)
    if args.fixture:
        payload["sample"] = True

    for path in (args.out, args.history_out):
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)
    out = full.dropna(subset=["composite"]).copy()
    out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
    out.to_csv(args.history_out, index=False)

    s = payload["stance"]
    print(f"As of {payload['as_of']}: {s['label']} (composite {s['composite']}, agreement {s['agreement']})")
    for f in payload["frameworks"]:
        print(f"  {f['vote_label']:9} {f['name']}: {f['reading']}")
    missing = [x["id"] for x in payload["sources"] if not x["available"]]
    if missing:
        print("Series with no data:", ", ".join(missing), file=sys.stderr)


if __name__ == "__main__":
    main()
