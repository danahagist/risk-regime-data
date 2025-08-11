# src/test_fred.py
from pandas_datareader import data as pdr

codes = ["NAPM","USSLIND","T10Y3M"]
for c in codes:
    df = pdr.DataReader(c, "fred", start="1990-01-01")
    print(c, "rows:", len(df))