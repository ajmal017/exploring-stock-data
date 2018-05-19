import pandas as pd
import datetime
from pandas_datareader import data as pdr

nifty500 = pd.read_csv('ind_nifty500list.csv')
symbols = nifty500.Symbol.tolist()

prices = []

for sym in symbols:
    print(sym)
    sym = sym.replace('-', '_')
    sym = sym.replace('&', '')
    data = pdr.DataReader('NSE/'+sym, 'quandl',
                          start=datetime.datetime(2005, 1, 1), 
                          end=datetime.datetime(2018, 6, 1))
    data.index.name = 'Date'
    data.reset_index(inplace=True)
    data['symbol'] = sym
    prices.append(data)

prices = pd.concat(prices, ignore_index=True)
prices = prices[['symbol'] + [c for c in prices.columns if c != 'symbol']]
prices.to_csv('prices.csv', index=False)
