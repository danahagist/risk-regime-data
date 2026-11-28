import json
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import brief  # noqa: E402
import compute_signals  # noqa: E402
import make_fixture  # noqa: E402


@pytest.fixture(scope="module")
def lessons():
    return json.load(open(os.path.join(ROOT, "docs", "data", "lessons.json")))


@pytest.fixture(scope="module")
def signals():
    payload, _ = compute_signals.build_payload(make_fixture.build("2026-11-27"), asof="2026-11-30")
    return json.loads(json.dumps(payload))      # exactly what the site and the brief read


def text_of(b):
    parts = [b["title"], b["summary"], b["disclaimer"]]
    for s in b["sections"]:
        parts += [s["heading"]] + s["paragraphs"] + s["bullets"]
    return "\n".join(parts)


def test_brief_has_every_section_and_reads_cleanly(signals, lessons):
    b = brief.build(signals, lessons)
    assert b["id"] == "2026-11" and signals["stance"]["label"] in b["title"]
    headings = [s["heading"] for s in b["sections"]]
    assert headings[:4] == ["The short version", "What changed since last month",
                            "Where the frameworks agree and disagree", "What this means for a portfolio"]
    assert headings[4].startswith("Framework in focus: ") and headings[5] == "What would change the picture"
    text = text_of(b)
    for bad in ("\u2014", "{", "}", " .", ".."):
        assert bad not in text, repr(bad)
    assert not re.search(r"\b(nan|None|null|undefined)\b", text)
    assert "not a forecast" in b["disclaimer"]


def test_brief_numbers_match_the_signals(signals, lessons):
    b = brief.build(signals, lessons)
    text = text_of(b)
    s, a = signals["stance"], signals["allocation"]
    assert f"{s['composite']:+.2f}" in text
    assert f"{a['model']['stocks']}% stocks" in text and f"{a['model']['cash']}% cash" in text
    scored = [f for f in signals["frameworks"] if f["vote"] is not None and f["family"] != "where"]
    assert f"Of the {len(scored)} frameworks" in text
    for change in signals["changes"]:
        assert change in text


def test_focus_rotates_and_points_at_a_live_lesson(signals, lessons):
    seen = set()
    for month in range(1, 13):
        s = dict(signals, as_of=f"2026-{month:02d}-28", as_of_label="x")
        focus = [x for x in brief.build(s, lessons)["sections"] if x["heading"].startswith("Framework in focus")][0]
        assert focus["link"] in {f["id"] for f in signals["frameworks"] if f["vote"] is not None}
        seen.add(focus["link"])
    assert len(seen) == 12                      # a different framework each month of the year


def test_next_month_wording_wraps_the_year():
    assert brief._next_month("2026-10-31") == ("November", "December")
    assert brief._next_month("2026-11-30") == ("December", "January")
    assert brief._next_month("2026-12-31") == ("January", "February")


def test_file_keeps_earlier_months_and_replaces_reruns(tmp_path, signals, lessons):
    path = str(tmp_path / "briefs.json")
    october = brief.build(dict(signals, as_of="2026-10-31", as_of_label="October 2026"), lessons)
    brief.update_file(october, path)
    brief.update_file(brief.build(signals, lessons), path)
    rerun = brief.build(signals, lessons)
    rerun["summary"] = "rerun"
    out = brief.update_file(rerun, path)
    assert [b["id"] for b in out] == ["2026-11", "2026-10"]         # newest first, no duplicates
    assert out[0]["summary"] == "rerun"
    assert json.load(open(path)) == out


def test_posts_index_points_at_real_files():
    posts = json.load(open(os.path.join(ROOT, "docs", "posts", "index.json")))
    for p in posts:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", p["date"])
        body = open(os.path.join(ROOT, "docs", "posts", p["file"])).read()
        assert len(body) > 200 and "—" not in body
