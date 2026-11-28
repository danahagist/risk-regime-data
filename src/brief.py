"""Write the monthly brief: a plain-language account of what the frameworks say.

The brief is assembled from the month's signals by fixed rules, so the same data
always produces the same text. It describes current readings. It does not forecast.

    python src/brief.py        # rebuild this month's brief from docs/data/signals.json
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

SIGNALS = "docs/data/signals.json"
LESSONS = "docs/data/lessons.json"
OUT_BRIEFS = "docs/data/briefs.json"
ASSET_NAME = {"stocks": "stocks", "bonds": "bonds", "cash": "cash", "alternatives": "alternatives"}
DISCLAIMER = ("This brief is assembled from the model's rules. It describes what well-known frameworks say about "
              "current conditions. It is not a forecast and it is not personal investment advice.")


def _signed(x, digits=2) -> str:
    if x is None:
        return "n/a"
    text = f"{x:+.{digits}f}"
    return text[1:] if float(text) == 0 else text       # plain 0.00, not +0.00 or -0.00


def _join(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _next_month(as_of: str) -> tuple[str, str]:
    d = datetime.strptime(as_of, "%Y-%m-%d")
    y, m = (d.year + (d.month // 12), d.month % 12 + 1)
    y2, m2 = (y + (m // 12), m % 12 + 1)
    return datetime(y, m, 1).strftime("%B"), datetime(y2, m2, 1).strftime("%B")


def _short_version(d: dict) -> dict:
    s = d["stance"]
    action = {
        "Overweight": f"That means holding {s['equity_offset']:.0f} points more in stocks than your neutral mix.",
        "Underweight": f"That means holding {abs(s['equity_offset'] or 0):.0f} points less in stocks than your neutral mix.",
        "Neutral": "That means holding your neutral mix. The frameworks do not agree strongly enough to lean either way.",
    }.get(s["label"], "There is not enough data this month to take a stance.")
    paras = [f"The consensus for {d['as_of_label']} is {s['label']}. {action}"]
    if s["composite"] is not None:
        alarms = ("No alarms are ringing." if not s["alarm_penalty"]
                  else f"The alarm families took {abs(s['alarm_penalty']):.2f} off.")
        paras.append(f"The score is {_signed(s['composite'])} on a scale from -1 to +1. It takes {_signed(s['threshold'], 1)} or more to lean "
                     f"into stocks and {_signed(-s['threshold'], 1)} or less to lean away. The driver families contributed "
                     f"{_signed(s['driver_score'])}. {alarms}")
    if s.get("raw_label") and s["raw_label"] != s["label"]:
        paras.append(f"This month's reading taken alone is {s['raw_label']}. A change needs two months in a row, so the stance has not moved yet.")
    for b in s["breakers"]:
        paras.append(b + ".")
    return {"heading": "The short version", "paragraphs": paras, "bullets": []}


def _changes(d: dict) -> dict:
    if d["changes"]:
        return {"heading": "What changed since last month", "paragraphs": [], "bullets": [c + "." for c in d["changes"]]}
    return {"heading": "What changed since last month",
            "paragraphs": ["Nothing. Every framework voted the same way as last month."], "bullets": []}


def _agreement(d: dict) -> dict:
    fams = {f["id"]: f for f in d["families"]}
    scored = [f for f in d["frameworks"] if fams[f["family"]]["role"] != "info" and f["vote"] is not None]
    pos = [f["name"] for f in scored if f["vote"] == 1]
    neu = [f["name"] for f in scored if f["vote"] == 0]
    neg = [f["name"] for f in scored if f["vote"] == -1]
    paras = [f"Of the {len(scored)} frameworks that feed the score, {len(pos)} are positive or clear, {len(neu)} are neutral or on watch, "
             f"and {len(neg)} are negative."]
    if neg:
        paras.append("The negatives: " + _join(neg) + ".")
    bullets = []
    for fam in d["families"]:
        members = [f for f in d["frameworks"] if f["family"] == fam["id"] and f["vote"] is not None]
        if not members:
            continue
        score = "not scored" if fam["role"] == "info" else _signed(fam["score"])
        bullets.append(f"{fam['name']} ({score}): " + "; ".join(m["reading"] for m in members) + ".")
    return {"heading": "Where the frameworks agree and disagree", "paragraphs": paras, "bullets": bullets}


def _portfolio(d: dict) -> dict:
    a, s = d["allocation"], d["stance"]
    mix = lambda m: ", ".join(f"{m[k]}% {ASSET_NAME[k]}" for k in ("stocks", "bonds", "cash", "alternatives"))  # noqa: E731
    paras = [f"For a neutral mix of {mix(a['neutral'])}, the rules give {mix(a['model'])} this month."]
    reasons = []
    if a["equity_offset"]:
        reasons.append(f"Stocks move {a['equity_offset']:+.0f} points because the consensus is {s['label']}.")
    else:
        reasons.append(f"Stocks stay at neutral because the consensus is {s['label']}.")
    if a["alt_tilt"] > 0:
        reasons.append(f"Alternatives get {a['alt_tilt']:.0f} extra points: {a['alt_note']}")
    elif a["alt_tilt"] < 0:
        reasons.append(f"Alternatives give up {abs(a['alt_tilt']):.0f} points: {a['alt_note']}")
    else:
        reasons.append(f"Alternatives stay at neutral: {a['alt_note']}")
    reasons.append({"cash": "The Fed is hiking, so half of the bond-and-cash sleeve sits in cash and T-bills.",
                    "bonds": "The Fed is cutting, so most of the bond-and-cash sleeve sits in bonds.",
                    "split": "The Fed is on hold, so bonds and cash keep their usual proportions."}[a["defensive"]])
    counts = {k: sum(1 for v in a["views"] if k in v["favors"]) for k in ASSET_NAME}
    paras.append("Counting each framework's own playbook: " + _join([f"{n} favor {ASSET_NAME[k]}" for k, n in counts.items()]) + ".")
    if s["valuation_tilt"] == "expensive":
        paras.append("Valuation reads expensive, so even an Overweight would be half the usual size.")
    return {"heading": "What this means for a portfolio", "paragraphs": paras, "bullets": reasons}


def _focus(d: dict, lessons: dict) -> dict | None:
    live = {f["id"]: f for f in d["frameworks"] if f["vote"] is not None}
    pool = [l for l in lessons["lessons"] if l["id"] in live]
    if not pool:
        return None
    when = datetime.strptime(d["as_of"], "%Y-%m-%d")
    lesson = pool[(when.year * 12 + when.month) % len(pool)]
    f = live[lesson["id"]]
    return {"heading": f"Framework in focus: {lesson['name']}",
            "paragraphs": [lesson["idea"], f"This month it reads {f['vote_label'].lower()}: {f['reading']}.",
                           "Where it has failed: " + lesson["fails"]],
            "bullets": [], "link": lesson["id"]}


def _watch(d: dict) -> dict:
    s = d["stance"]
    paras = []
    if s["composite"] is not None:
        up, down = s["threshold"] - s["composite"], s["composite"] + s["threshold"]
        if s["label"] == "Neutral":
            paras.append(f"The score would need to rise {up:.2f} to reach Overweight or fall {down:.2f} to reach Underweight, "
                         "and hold there for two months.")
        elif s["label"] == "Overweight":
            paras.append(f"The score has {-up:.2f} of room before it falls back to Neutral territory.")
        elif s["label"] == "Underweight":
            paras.append(f"The score is {-down:.2f} below the line where it would return to Neutral territory.")
    fams = {f["id"]: f for f in d["families"]}
    alarms = [f for f in d["frameworks"] if fams[f["family"]]["role"] == "alarm" and f["vote"] is not None]
    watching = [f["name"] for f in alarms if f["vote"] == 0]
    ringing = [f["name"] for f in alarms if f["vote"] == -1]
    if ringing:
        paras.append("Alarms triggered: " + _join(ringing) + ".")
    if watching:
        paras.append("Alarms on watch: " + _join(watching) + ". A move to triggered would pull the score down.")
    if not ringing and not watching and alarms:
        paras.append("All of the recession and credit alarms are clear.")
    nxt, after = _next_month(d["as_of"])
    paras.append(f"The next brief covers {nxt} and arrives in the first days of {after}.")
    return {"heading": "What would change the picture", "paragraphs": paras, "bullets": []}


def build(signals: dict, lessons: dict) -> dict:
    s = signals["stance"]
    sections = [_short_version(signals), _changes(signals), _agreement(signals), _portfolio(signals)]
    focus = _focus(signals, lessons)
    if focus:
        sections.append(focus)
    sections.append(_watch(signals))
    n_scored = sum(1 for f in signals["frameworks"] if f["vote"] is not None)
    return {
        "id": signals["as_of"][:7],
        "date": signals["as_of"],
        "title": f"{signals['as_of_label']}: the frameworks say {s['label']}",
        "summary": f"{n_scored} frameworks, one stance: {s['label']}, with a score of {_signed(s['composite'])}.",
        "sections": sections,
        "disclaimer": DISCLAIMER,
        "sample": bool(signals.get("sample")),
    }


def update_file(brief: dict, path: str = OUT_BRIEFS) -> list[dict]:
    """Add or replace this month's brief, keeping earlier months."""
    briefs = []
    if os.path.exists(path):
        with open(path) as fh:
            briefs = json.load(fh)
    briefs = [b for b in briefs if b["id"] != brief["id"]] + [brief]
    briefs.sort(key=lambda b: b["id"], reverse=True)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(briefs, fh, indent=1, ensure_ascii=False)
    return briefs


if __name__ == "__main__":
    with open(SIGNALS) as fh:
        signals = json.load(fh)
    with open(LESSONS) as fh:
        lessons = json.load(fh)
    if signals.get("sample"):
        print("signals.json holds sample data; not writing a brief from it.", file=sys.stderr)
        sys.exit(1)
    out = update_file(build(signals, lessons))
    print(f"Wrote {OUT_BRIEFS}: {len(out)} brief(s), latest {out[0]['title']}")
