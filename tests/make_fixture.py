"""Write SYNTHETIC sample series for tests and offline previews.

These are not real data. Paths are interpolated between rough anchor points with
noise added, so pages rendered from them look plausible but must be labeled as
sample data.

    python tests/make_fixture.py OUT_DIR [--end 2026-11-27]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def path(anchors: dict, freq: str, end: str, noise: float = 0.0, log: bool = False, seed: int = 0) -> pd.Series:
    """Interpolate between dated anchor values on a regular calendar, plus noise."""
    a = pd.Series(anchors, dtype="float64")
    a.index = pd.to_datetime(a.index)
    idx = pd.date_range(a.index.min(), pd.Timestamp(end), freq=freq)
    vals = np.log(a) if log else a
    s = vals.reindex(vals.index.union(idx)).interpolate(method="time").ffill().reindex(idx)
    rng = np.random.default_rng(seed)
    s = s + rng.normal(0, noise, len(s))
    return np.exp(s) if log else s


def build(end: str = "2026-11-27") -> dict[str, pd.Series]:
    B, W, MS, QS = "B", "W-SAT", "MS", "QS"
    s: dict[str, pd.Series] = {}
    s["SP500"] = path({"2012-01-02": 1280, "2016-01-04": 1940, "2018-01-02": 2820, "2018-12-24": 2350,
                       "2020-02-19": 3390, "2020-03-23": 2240, "2021-12-31": 4770, "2022-10-12": 3580,
                       "2023-12-29": 4770, "2024-12-31": 5880, "2025-04-08": 4980, "2025-12-31": 6845,
                       "2026-02-13": 7150, "2026-03-20": 6600, "2026-05-29": 7350, "2026-07-31": 7600,
                       "2026-09-30": 7700, "2026-10-06": 7819, "2026-11-04": 7690,
                       "2026-11-20": 6910, "2026-11-27": 7298}, B, end, 0.004, log=True, seed=1)
    s["VIXCLS"] = path({"2012-01-02": 20, "2017-06-01": 11, "2018-12-24": 30, "2020-01-02": 13,
                        "2020-03-16": 75, "2020-12-31": 23, "2021-12-31": 18, "2022-10-12": 33,
                        "2023-12-29": 13, "2025-04-08": 45, "2025-12-31": 15, "2026-10-06": 16,
                        "2026-11-04": 20, "2026-11-13": 21, "2026-11-20": 36, "2026-11-27": 23.8}, B, end, 0.6, seed=2).clip(lower=9)
    s["VXVCLS"] = (s["VIXCLS"] * 0.85 + 3.6).rename(None)
    s["BAMLH0A0HYM2"] = path({"2023-11-27": 3.9, "2024-12-31": 2.9, "2025-04-08": 4.5, "2025-12-31": 2.9,
                              "2026-09-30": 2.7, "2026-11-20": 3.67, "2026-11-27": 3.45}, B, end, 0.03, seed=3)
    s["DGS10"] = path({"2012-01-02": 1.95, "2018-10-05": 3.2, "2020-08-04": 0.55, "2022-10-21": 4.2,
                       "2023-10-19": 4.98, "2024-09-16": 3.65, "2025-12-31": 4.2, "2026-10-05": 5.31,
                       "2026-11-27": 4.94}, B, end, 0.02, seed=4)
    s["DGS2"] = path({"2012-01-02": 0.25, "2018-11-08": 2.95, "2020-08-04": 0.12, "2023-10-18": 5.2,
                      "2024-09-16": 3.6, "2025-12-31": 3.6, "2026-10-07": 4.92, "2026-11-27": 4.69},
                     B, end, 0.02, seed=5)
    s["DGS3MO"] = path({"2012-01-02": 0.03, "2015-12-01": 0.2, "2019-03-01": 2.45, "2020-04-01": 0.1,
                        "2022-03-01": 0.4, "2023-08-01": 5.45, "2024-12-31": 4.3, "2025-12-31": 3.6,
                        "2026-09-16": 3.85, "2026-11-27": 3.95}, B, end, 0.01, seed=6).clip(lower=0.01)
    s["DFEDTARU"] = path({"2012-01-02": 0.25, "2015-12-16": 0.25, "2018-12-20": 2.5, "2019-07-31": 2.5,
                          "2019-10-31": 1.75, "2020-03-15": 1.75, "2020-03-16": 0.25, "2022-03-16": 0.25,
                          "2023-07-27": 5.5, "2024-09-18": 5.5, "2024-12-19": 4.5, "2025-09-17": 4.5,
                          "2025-12-11": 3.75, "2026-09-15": 3.75, "2026-09-16": 4.0}, "D", end)
    s["DFEDTARU"] = (s["DFEDTARU"] * 4).round() / 4
    s["FEDFUNDS"] = (s["DFEDTARU"].resample(MS).mean() - 0.12).clip(lower=0.05)
    s["NFCI"] = path({"2012-01-07": -0.45, "2020-01-04": -0.75, "2020-03-28": 0.3, "2021-06-05": -0.7,
                      "2022-10-08": -0.1, "2024-12-28": -0.6, "2026-09-26": -0.5, "2026-11-21": -0.38},
                     W, end, 0.01, seed=7)
    s["NFCICREDIT"] = (s["NFCI"] * 0.6 - 0.05).rename(None)
    s["ICSA"] = path({"2012-01-07": 370000, "2019-12-28": 218000, "2020-04-04": 6100000,
                      "2020-12-26": 800000, "2021-12-25": 205000, "2024-12-28": 220000,
                      "2026-11-21": 203000}, W, end, 0.02, log=True, seed=8)
    s["CFNAI"] = path({"2012-01-01": 0.1, "2019-12-01": -0.2, "2020-04-01": -17.0, "2020-06-01": 4.0,
                       "2021-06-01": 0.4, "2022-12-01": -0.3, "2024-12-01": -0.1, "2026-03-01": -0.15,
                       "2026-06-01": 0.15, "2026-10-01": 0.22}, MS, end, 0.12, seed=9)
    s["INDPRO"] = path({"2012-01-01": 96, "2019-12-01": 101.6, "2020-04-01": 84.6, "2021-12-01": 100.6,
                        "2024-12-01": 101.2, "2025-10-01": 101.2, "2026-10-01": 104.1}, MS, end, 0.1, seed=10)
    s["UNRATE"] = path({"2012-01-01": 8.3, "2019-12-01": 3.6, "2020-04-01": 14.8, "2021-12-01": 3.9,
                        "2023-04-01": 3.4, "2024-07-01": 4.2, "2025-12-01": 4.4, "2026-03-01": 4.3,
                        "2026-10-01": 4.1}, MS, end).round(1)
    low12 = s["UNRATE"].rolling(3).mean().rolling(12).min()
    s["SAHMREALTIME"] = (s["UNRATE"].rolling(3).mean() - low12.shift(1)).clip(lower=0).dropna().round(2)
    s["PCEPILFE"] = path({"2011-01-01": 91.0, "2020-12-01": 106.6, "2022-12-01": 117.6, "2024-12-01": 124.5,
                          "2025-10-01": 127.4, "2026-04-01": 129.3, "2026-10-01": 131.6}, MS, end, log=True)
    s["CPIAUCSL"] = path({"2011-01-01": 221.0, "2020-12-01": 262.0, "2022-12-01": 298.8, "2024-12-01": 317.6,
                          "2025-10-01": 325.0, "2026-10-01": 337.0}, MS, end, log=True)
    s["NROU"] = path({"2010-01-01": 5.0, "2020-01-01": 4.4, "2030-01-01": 4.3}, QS, "2030-01-01")
    s["GDP"] = path({"1970-01-01": 1050, "1990-01-01": 5870, "2000-01-01": 10000, "2010-01-01": 14760,
                     "2020-01-01": 21730, "2026-07-01": 31900}, QS, end, log=True)
    ratio = path({"1970-01-01": 0.62, "1982-07-01": 0.33, "2000-01-01": 1.40, "2002-10-01": 0.75,
                  "2007-07-01": 1.05, "2009-01-01": 0.57, "2021-10-01": 1.95, "2022-10-01": 1.50,
                  "2026-07-01": 2.15}, QS, end, log=True)
    s["NCBEILQ027S"] = ratio * s["GDP"] * 1000.0
    # ---- ETF prices: the S&P path scaled by a per-fund drift, plus noise
    spx = s["SP500"]
    def etf(seed, drift, vol=0.004, base=None):
        rng = np.random.default_rng(seed)
        b = spx if base is None else base
        days = np.arange(len(b))
        return (b * np.exp(drift * days / 252.0 + rng.normal(0, vol, len(b)).cumsum() * 0.3)).rename(None)
    flat = pd.Series(100.0, index=spx.index)
    s["PX:SPY"] = spx / 10.0
    s["PX:ACWX"] = etf(21, -0.03)
    s["PX:IWM"] = etf(22, -0.02)
    s["PX:RSP"] = etf(23, -0.01)
    s["PX:DIA"] = etf(24, -0.01)
    s["PX:IYT"] = etf(25, -0.03)
    s["PX:AGG"] = (100 - (s["DGS10"].reindex(spx.index).ffill() - 2.0) * 6.0).rename(None)
    s["PX:BIL"] = (90 * np.exp((s["DGS3MO"].reindex(spx.index).ffill() / 100 / 252).cumsum())).rename(None)
    s["PX:GLD"] = path({"2012-01-02": 160, "2015-12-01": 102, "2020-08-06": 194, "2022-10-03": 155, "2025-12-31": 400,
                        "2026-06-01": 440, "2026-11-27": 338}, B, end, 0.004, log=True, seed=26)
    s["PX:DBC"] = path({"2012-01-02": 28, "2016-01-20": 12, "2018-10-03": 18, "2020-04-21": 10.5, "2022-06-08": 30,
                        "2025-12-31": 23, "2026-10-30": 29, "2026-11-27": 25}, B, end, 0.005, log=True, seed=27)
    s["PX:CPER"] = path({"2012-01-02": 25, "2016-01-15": 13, "2021-05-10": 29, "2025-12-31": 30, "2026-11-27": 31}, B, end, 0.006, log=True, seed=28)
    for i, (t, drift) in enumerate({"XLK": 0.05, "XLC": 0.01, "XLY": 0.0, "XLF": -0.01, "XLI": -0.01, "XLB": -0.03,
                                     "XLE": -0.04, "XLV": -0.02, "XLP": -0.04, "XLU": -0.04, "XLRE": -0.05}.items()):
        s[f"PX:{t}"] = etf(40 + i, drift, 0.006)
    # ---- valuation history. Like the real file, it stops in 2023; later months are estimated.
    hist_end = "2023-09-01"
    s["SHILLER:PRICE"] = path({"1900-01-01": 6.1, "1929-09-01": 31.3, "1932-06-01": 4.8, "1966-01-01": 93, "1982-07-01": 109,
                               "1999-12-01": 1428, "2009-03-01": 757, "2021-12-01": 4675, "2022-10-01": 3726, "2023-09-01": 4409}, MS, hist_end, log=True)
    s["SHILLER:EPS"] = path({"1900-01-01": 0.48, "1929-09-01": 1.6, "1932-06-01": 0.5, "1966-01-01": 5.2, "1982-07-01": 14,
                             "1999-12-01": 48, "2009-03-01": 7, "2010-12-01": 77, "2021-12-01": 198, "2023-06-01": 181}, MS, "2023-06-01", log=True)
    s["SHILLER:CPI"] = path({"1900-01-01": 7.9, "1920-06-01": 20.9, "1933-01-01": 12.9, "1966-01-01": 31.8, "1982-07-01": 97.5,
                             "1999-12-01": 168.3, "2009-03-01": 212.7, "2021-12-01": 278.8, "2023-09-01": 307.8}, MS, hist_end, log=True)
    s["SHILLER:CAPE"] = path({"1900-01-01": 18, "1921-01-01": 5.5, "1929-09-01": 32, "1932-06-01": 6, "1966-01-01": 24,
                              "1982-07-01": 6.6, "1999-12-01": 44, "2009-03-01": 13.3, "2021-11-01": 38.6, "2022-10-01": 27,
                              "2023-09-01": 30.8}, MS, hist_end, log=True)
    s["SHILLER:PE"] = (s["SHILLER:PRICE"] / s["SHILLER:EPS"]).dropna()
    s["VAL:EPS_TTM"] = pd.Series([292.0], index=[pd.Timestamp(end)])     # implied by a fund P/E of about 25
    s["DFII10"] = path({"2012-01-02": -0.1, "2013-12-31": 0.8, "2021-08-03": -1.15, "2023-10-25": 2.5, "2025-12-31": 2.0,
                        "2026-11-27": 2.9}, B, end, 0.01, seed=29)
    # ---- sentiment and liquidity
    s["AAII:SPREAD"] = path({"2012-01-05": 10, "2020-03-19": -17, "2021-04-08": 34, "2022-09-22": -43, "2024-07-18": 30,
                             "2025-04-03": -40, "2026-09-24": 5, "2026-11-25": -8.3}, "W-THU", end, 6.0, seed=30)
    s["WALCL"] = path({"2012-01-04": 2.9e6, "2015-01-07": 4.5e6, "2019-09-04": 3.76e6, "2020-06-10": 7.17e6,
                       "2022-04-13": 8.96e6, "2025-12-31": 6.6e6, "2026-11-25": 6.45e6}, "W-WED", end)   # millions
    s["WTREGEN"] = path({"2012-01-04": 60, "2020-07-22": 1800, "2021-12-29": 400, "2023-06-07": 50, "2025-12-31": 800,
                         "2026-11-25": 850}, "W-WED", end)                                                 # billions
    s["RRPONTSYD"] = path({"2013-09-23": 5, "2021-03-17": 1, "2022-12-30": 2550, "2024-12-31": 470, "2025-12-31": 60,
                           "2026-11-27": 40}, B, end).clip(lower=0)                                        # billions
    return {k: v.dropna() for k, v in s.items()}


def write(directory: str, end: str = "2026-11-27") -> None:
    os.makedirs(directory, exist_ok=True)
    for code, series in build(end).items():
        pd.DataFrame({"date": series.index, "value": series.values}).to_csv(
            os.path.join(directory, f"{code.replace(':', '_')}.csv"), index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--end", default="2026-11-27")
    a = ap.parse_args()
    write(a.out, a.end)
    print(f"Wrote synthetic sample series to {a.out}")
