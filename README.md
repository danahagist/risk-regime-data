# Market Regime Consensus

A monthly dashboard that collects key economic and market metrics, runs them through well-known investing frameworks, and combines the results into one consensus stance and a model allocation across stocks, bonds, cash and alternatives. Every framework and metric comes with a plain-English lesson.

Educational model output. Not personal investment advice.

## What is in here

| Path | What it does |
|---|---|
| `src/fred_data.py` | The FRED series catalog, with a `public` or `third_party` rights flag on each |
| `src/market_data.py` | ETF prices, long-run valuation series and the AAII survey, each with a fallback and a safe failure mode |
| `src/frameworks.py` | Framework votes, family scoring, the consensus rules and the model allocation. Pure functions, no network |
| `src/compute_signals.py` | Fetches data, runs the frameworks, writes `docs/data/signals.json` and `data/consensus_history.csv` |
| `src/compute_macro_monthly.py` | The original macro and risk regime script, feeding the classic view |
| `src/brief.py` | Writes the monthly brief from the signals, by fixed rules, into `docs/data/briefs.json` |
| `docs/posts/` | Your own posts, as Markdown. Add a file, then list it in `docs/posts/index.json` |
| `src/backtest.py` | What following each framework, and the consensus allocation, would have returned |
| `docs/` | The site (GitHub Pages): dashboard, lessons, calculator, briefs, backtest, methodology, classic view |
| `docs/data/lessons.json` | Lesson text for every framework and metric. Edit this to change the wording |
| `tests/` | Tests for the scoring rules, run on synthetic series |

## Running it

```bash
pip install -r requirements.txt
export FRED_API_KEY=...            # free at fred.stlouisfed.org
python src/compute_macro_monthly.py
python src/compute_signals.py
```

The GitHub Action runs both on the 2nd of each month and commits the refreshed data. You can also start it by hand from the Actions tab.

To preview the site without a key, on synthetic sample data:

```bash
pip install -r requirements-dev.txt
python tests/make_fixture.py /tmp/fx
python src/compute_signals.py --fixture /tmp/fx --asof 2026-11-30
python -m http.server -d docs 8000   # the page shows a "sample data" banner
```

Do not commit a `signals.json` built from the fixture.

## Tests

```bash
python -m pytest
```

## How the consensus works

See `docs/methodology.html` for the full rules. In brief: each framework votes monthly, votes are averaged within families, four driver families set direction, two alarm families can only subtract, valuation sets the size of a move, and a change needs two months in a row.

The thresholds and weights at the top of `src/frameworks.py` were set once from the logic of each framework. Please do not tune them to a backtest.

## Data sources and what happens when one fails

| Data | Source | Fallback |
|---|---|---|
| Economic series, yields, spreads, VIX, S&P 500 | FRED API | none needed |
| ETF prices (22 funds) | Yahoo Finance via `yfinance` | Stooq |
| Shiller CAPE and P/E history | GitHub mirror of Shiller's dataset (ends 2023) | Yale's `ie_data.xls` |
| Current P/E | Yahoo Finance fund data, used to estimate months after 2023 | multpl.com when it is reachable |
| Investor sentiment | AAII survey file | none |

Outside sources sometimes block automated requests. When one fails, the run still succeeds: the frameworks that needed it show "No data" and are left out of the averages. The workflow log lists which sources and frameworks came back empty.

## Data rights

Series produced by government agencies or Federal Reserve banks can be shown publicly. Everything else is owned by a third party and is fine for personal use only: the S&P 500, the VIX and 3-month VIX, the ICE BofA high-yield spread, ETF prices, the valuation history and the AAII survey. Before a public or commercial launch, replace them with a licensed feed or public substitutes. `src/fred_data.py` and `src/market_data.py` mark which is which, and the methodology page lists them.

FRED's API terms also require the notice shown in the site footer.

## Roadmap

- Trend and momentum for more asset classes (real estate, TIPS, international bonds).
- Company fundamentals from SEC EDGAR, then sector and security-level views.
- A licensed price feed, if the site goes public.
