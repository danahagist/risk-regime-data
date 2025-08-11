# src/compute_risk.py
import time
import pandas as pd
import numpy as np
import yfinance as yf

# ---- Output CSV lives in REPO ROOT (not /data) ----
OUT_PATH = "risk_weekly_history.csv"

# Tickers we need. ^VXV is a fallback for 3M VIX if ^VIX3M fails.
TICKERS = ['SPY', 'HYG', 'LQD', '^VIX', '^VIX3M', '^VXV']

# ----------------- Helpers -----------------
def ma(s, n):
    return s.rolling(n, min_periods=max(5, n // 5)).mean()

def zscore(s, n):
    m = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (s - m) / sd

def download_prices(tickers, tries=3, sleep=6):
    """
    Robust fetch for GitHub Actions:
      1) Attempt batch download with threads disabled
      2) Fill missing tickers with per-ticker .history()
      3) Retry a few times before giving up
    Returns: wide DataFrame of Close prices (daily)
    """
    last_err = None
    for attempt in range(1, tries + 1):
        try:
            df = yf.download(
                tickers,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=False,      # reduce flakiness in CI
                actions=False,
            )
            # Normalize to wide Close frame
            if isinstance(df.columns, pd.MultiIndex):
                df = df.xs("Close", axis=1, level=1)
            df = df.dropna(how="all")

            # Fill any missing tickers with per-ticker fetch
            missing = [t for t in tickers if t not in df.columns or df[t].dropna().empty]
            for t in missing:
                try:
                    h = yf.Ticker(t).history(period="max", interval="1d", auto_adjust=True)
                    if not h.empty and "Close" in h:
                        df[t] = h["Close"]
                        print(f"Filled {t} via per-ticker history()")
                except Exception as e2:
                    last_err = e2

            # Require at least SPY to proceed
            if "SPY" in df.columns and not df["SPY"].dropna().empty:
                print(f"Download attempt {attempt}: success")
                return df.sort_index()

            last_err = last_err or RuntimeError("Empty after batch+fallback")
            print(f"Download attempt {attempt}: incomplete, retrying...")
        except Exception as e:
            last_err = e
            print(f"Download attempt {attempt} failed: {e}")
        time.sleep(sleep)
    raise last_err

# ----------------- Main -----------------
def main():
    # 1) Download daily prices
    try:
        px = download_prices(TICKERS, tries=3, sleep=6)
    except Exception as e:
        print("WARN: price download failed:", repr(e))
        print("Skipping update this run (will try again next schedule).")
        return  # safe exit

    if px.empty:
        print("WARN: got empty price frame. Skipping.")
        return

    # 2) Resample to weekly (Friday)
    px_w = px.resample('W-FRI').last().dropna(how='all')

    # 3) Fallback for 3M VIX (^VIX3M); use ^VXV if present
    if '^VIX3M' not in px_w.columns or px_w['^VIX3M'].dropna().empty:
        if '^VXV' in px_w.columns and not px_w['^VXV'].dropna().empty:
            px_w['^VIX3M'] = px_w['^VXV']
            print("Used ^VXV as fallback for ^VIX3M.")
        else:
            print("WARN: missing both ^VIX3M and ^VXV — cannot compute VIX term structure. Skipping.")
            return

    # 4) Sanity checks for required series
    for name in ['SPY', 'HYG', 'LQD', '^VIX', '^VIX3M']:
        if name not in px_w.columns or px_w[name].dropna().empty:
            print(f"WARN: missing or empty series {name}. Skipping.")
            return

    # 5) Signals (same logic you designed)
    spy = px_w['SPY']
    hyg = px_w['HYG']
    lqd = px_w['LQD']
    vix = px_w['^VIX']
    vix3m = px_w['^VIX3M']

    # Trend: SPY > ~200d MA -> 40 weeks
    trend_on = (spy > ma(spy, 40)).astype(int)

    # Credit: HYG/LQD ratio with ~180d z-score -> 36 weeks, and below 10w MA
    ratio = (hyg / lqd)
    ratio_z180 = zscore(ratio, 36)
    ratio_ma10 = ma(ratio, 10)
    credit_off = ((ratio_z180 < -1.0) & (ratio < ratio_ma10)).astype(int)

    # Vol: VIX term structure (backwardation if VIX3M - VIX < 0)
    term = (vix3m - vix)
    vix_off = (term < 0).astype(int)

    signals = pd.DataFrame({
        'trend_on': trend_on,
        'credit_off': credit_off,
        'vix_off': vix_off
    }).dropna()

    if signals.empty:
        print("WARN: signals frame empty after dropna. Skipping.")
        return

    off_votes = (1 - signals['trend_on']) + signals['credit_off'] + signals['vix_off']
    regime_raw = np.where(off_votes >= 2, 'Off', 'On')

    # Sustained-change filter: require 2 consecutive weeks to flip
    regime = pd.Series(regime_raw, index=signals.index, name='regime')
    final = regime.copy()
    for i in range(1, len(final)):
        if final.iloc[i] != final.iloc[i - 1]:
            if i + 1 < len(final) and final.iloc[i + 1] != final.iloc[i]:
                final.iloc[i] = final.iloc[i - 1]

    out = pd.concat([signals, off_votes.rename('off_votes'), final.rename('regime')], axis=1)

    if out.empty:
        print("WARN: no rows in output; skipping this run.")
        return

    # 6) Append last week to CSV in repo root
    last = out.iloc[-1]
    row = pd.DataFrame([{
        "date": out.index[-1].date().isoformat(),
        "trend_on": int(last['trend_on']),
        "credit_off": int(last['credit_off']),
        "vix_off": int(last['vix_off']),
        "off_votes": int(last['off_votes']),
        "regime": last['regime']
    }])

    try:
        hist = pd.read_csv(OUT_PATH, parse_dates=['date'])
    except FileNotFoundError:
        hist = pd.DataFrame(columns=row.columns)

    new_dt = pd.to_datetime(row.iloc[0]['date']).date()
    if not (len(hist) and (hist['date'].dt.date == new_dt).any()):
        hist = pd.concat([hist, row], ignore_index=True)
        hist.to_csv(OUT_PATH, index=False)
        print("Appended:", row.to_dict(orient="records")[0])
    else:
        print("Already have latest week:", row.iloc[0]['date'])

if __name__ == "__main__":
    main()