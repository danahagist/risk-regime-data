import yfinance as yf

tickers = ["^VIX", "^VIX3M", "SPY", "HYG", "LQD"]
print("Testing Yahoo Finance data fetch...")

for t in tickers:
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False)
        print(f"{t}: {len(df)} rows")
        print(df.tail(1))  # last row for quick check
    except Exception as e:
        print(f"❌ Failed to get {t}: {e}")
        
import yfinance as yf

tickers = ["^VIX", "^VIX3M", "SPY", "HYG", "LQD"]
print("Testing Yahoo Finance (5d)…")
ok = True
for t in tickers:
    try:
        df = yf.download(t, period="5d", interval="1d", progress=False, auto_adjust=True, threads=False)
        print(f"{t}: {len(df)} rows")
        print(df.tail(1))
        ok &= len(df) > 0
    except Exception as e:
        print(f"{t} error: {e}")
        ok = False

# optional: fail fast if nothing returned
import sys; sys.exit(0 if ok else 2)