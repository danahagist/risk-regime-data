# # src/test_fred.py
from pandas_datareader import data as pdr
for c in ["NAPM","USSLIND","T10Y3M"]:
    df = pdr.DataReader(c, "fred", start="1990-01-01")
    print(c, "rows:", len(df))