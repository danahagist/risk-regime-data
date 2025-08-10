import pandas as pd, numpy as np
import yfinance as yf

def ma(s, n): return s.rolling(n, min_periods=max(5, n//5)).mean()
def zscore(s, n):
    m=s.rolling(n).mean(); sd=s.rolling(n).std()
    return (s-m)/sd

def main():
    tickers=['SPY','HYG','LQD','^VIX','^VIX3M','^VXV']  # ^VXV as 3m VIX fallback
    px=yf.download(tickers, period='max', auto_adjust=True, progress=False)['Close'].dropna(how='all')
    px_w=px.resample('W-FRI').last().dropna(how='all')
    if '^VIX3M' not in px_w.columns or px_w['^VIX3M'].isna().all():
        px_w['^VIX3M']=px_w['^VXV']

    spy, hyg, lqd = px_w['SPY'], px_w['HYG'], px_w['LQD']
    vix, vix3m    = px_w['^VIX'], px_w['^VIX3M']

    # 1) Trend (SPY vs ~200d MA => 40 weeks)
    spy_ma=ma(spy, 40)
    trend_on=(spy>spy_ma).astype(int)

    # 2) Credit (HYG/LQD ratio z~180d => 36 weeks + below 10w MA)
    ratio=(hyg/lqd)
    ratio_z=zscore(ratio, 36)
    ratio_ma10=ma(ratio,10)
    credit_off=((ratio_z<-1.0) & (ratio<ratio_ma10)).astype(int)

    # 3) Vol (VIX term structure)
    term=(vix3m - vix)
    vix_off=(term<0).astype(int)

    signals=pd.DataFrame({'trend_on':trend_on,'credit_off':credit_off,'vix_off':vix_off}).dropna()
    off_votes=(1-signals['trend_on'])+signals['credit_off']+signals['vix_off']
    regime_raw=np.where(off_votes>=2,'Off','On')

    # sustained-change filter (require 2 consecutive weeks to flip)
    regime=pd.Series(regime_raw,index=signals.index,name='regime')
    final=regime.copy()
    for i in range(1,len(final)):
        if final.iloc[i]!=final.iloc[i-1]:
            if i+1<len(final) and final.iloc[i+1]!=final.iloc[i]:
                final.iloc[i]=final.iloc[i-1]

    out=pd.concat([signals, off_votes.rename('off_votes'), final.rename('regime')], axis=1)
    last=out.iloc[-1]
    row=pd.DataFrame([{
        "date": out.index[-1].date().isoformat(),
        "trend_on": int(last['trend_on']),
        "credit_off": int(last['credit_off']),
        "vix_off": int(last['vix_off']),
        "off_votes": int(last['off_votes']),
        "regime": last['regime']
    }])

    import os
    os.makedirs('data', exist_ok=True)
    path='risk_weekly_history.csv'  # your file is in repo root (not data/)
    try:
        hist=pd.read_csv(path, parse_dates=['date'])
    except FileNotFoundError:
        hist=pd.DataFrame(columns=row.columns)

    # append latest week if missing
    if not (len(hist) and (hist['date'].dt.date==pd.to_datetime(row.iloc[0]['date']).date()).any()):
        hist=pd.concat([hist,row], ignore_index=True)
        hist.to_csv(path,index=False)
        print("Appended:", row.to_dict(orient="records")[0])
    else:
        print("Already have latest week:", row.iloc[0]['date'])

if __name__=="__main__":
    main()