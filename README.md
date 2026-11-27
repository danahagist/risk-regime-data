# Market Regime Consensus

A monthly dashboard that collects key economic and market metrics, runs them through well-known investing frameworks, and combines the results into one consensus stance and a model allocation across stocks, bonds, cash and alternatives. Every framework and metric comes with a plain-English lesson.

Educational model output. Not personal investment advice.

## What is in here

| Path | What it does |
|---|---|
| `src/fred_data.py` | The FRED series catalog, with a `public` or `third_party` rights flag on each |
| `src/frameworks.py` | Framework votes, family scoring, the consensus rules and the model allocation. Pure functions, no network |
| `src/compute_signals.py` | Fetches data, runs the frameworks, writes `docs/data/signals.json` and `data/consensus_history.csv` |
| `src/compute_macro_monthly.py` | The original macro and risk regime script, feeding the classic view |
| `docs/` | The site (GitHub Pages): dashboard, lessons, calculator, methodology, classic view |
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

## Data rights

Most series are produced by government agencies or Federal Reserve banks and can be shown publicly. Four market series relayed by FRED are owned by third parties: the S&P 500, the VIX, the 3-month VIX and the ICE BofA high-yield spread. They are fine for personal use. Before a public or commercial launch, replace them with a licensed feed or public substitutes (ETF prices from a licensed vendor, the Chicago Fed credit subindex, realized volatility). `src/fred_data.py` marks which is which, and the methodology page lists them.

FRED's API terms also require the notice shown in the site footer.

## Roadmap

- **Phase 2, price data:** dual momentum, breadth, sector rotation, intermarket ratios, trend for other asset classes.
- **Phase 3, outside files:** Shiller CAPE and excess CAPE yield, Fed model, Rule of 20, AAII sentiment, a liquidity gauge.
- **Later:** company fundamentals from SEC EDGAR, sector and security-level views.

Planned frameworks already have lessons on the Learn page, marked "Coming soon".
