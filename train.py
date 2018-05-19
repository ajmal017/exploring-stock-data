#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 21:43:33 2018

@author: subhajit
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np

prices = pd.read_csv('prices.csv', parse_dates=[1])
ratios = pd.read_csv('ratios.csv', parse_dates=[1])
mktidx = pd.read_csv('nifty500hist.csv', parse_dates=[0])
stockdetail = pd.read_csv('ind_nifty500list.csv')
stockdetail = stockdetail[['Symbol', 'Industry']]
return_window = 250

# adjusting for splits
prices.sort_values(['symbol', 'Date'], ascending=[True, False], inplace=True)
prices['num_split'] = round(prices['High']/prices.groupby('symbol')['High'].shift(1), 0)
prices['num_split'].fillna(1, inplace=True)
prices['split_factor'] = 1/prices.groupby('symbol')['num_split'].cumprod()
prices['adj_open'] = prices['Open'] * prices['split_factor']
prices['adj_high'] = prices['High'] * prices['split_factor']
prices['adj_low'] = prices['Low'] * prices['split_factor']
prices['adj_close'] = prices['Close'] * prices['split_factor']

temp = ratios.merge(prices, how='outer', left_on=['symbol', 'date'], right_on=['symbol', 'Date'])
temp.Date.fillna(temp.date, inplace=True)
temp.num_split.fillna(1, inplace=True)
temp.sort_values(['symbol', 'Date'], ascending=[True, False], inplace=True)
temp['split_factor'] = 1/temp.groupby('symbol')['num_split'].cumprod()
temp['adj_eps'] = temp['Diluted EPS (Rs.)'] * temp['split_factor']
temp['adj_bookvalue'] = temp['Book Value [Excl. Reval Reserve]/Share (Rs.)'] * temp['split_factor']
temp = temp[temp.date.notnull()][['symbol', 'date', 'adj_eps', 'adj_bookvalue']]
ratios = ratios.merge(temp, on=['symbol', 'date'])
del temp

ratios.sort_values(['symbol', 'date'], inplace=True)
ratios['eps_growth'] = 100*ratios.groupby('symbol')['adj_eps'].pct_change(1).values

prices.drop(['num_split', 'split_factor'], axis=1, inplace=True)

# creating variables for analysis
prices.sort_values(['symbol', 'Date'], inplace=True)
prices['returns'] = 100*prices.groupby('symbol')['adj_close'].pct_change(return_window).shift(-return_window).values
prices['up_from_52week_low'] = 100*prices['adj_close']/prices.groupby('symbol')['adj_low'].apply(lambda x: pd.rolling_min(x, window=250)) - 100
prices['down_from_52week_high'] = 100 - 100*prices['adj_close']/prices.groupby('symbol')['adj_high'].apply(lambda x: pd.rolling_max(x, window=250))
prices['daily_returns'] = 100*prices.groupby('symbol')['adj_close'].pct_change(1).values
prices['volatility'] = prices.groupby('symbol')['daily_returns'].apply(lambda x: pd.rolling_std(x, window=250))
prices.dropna(inplace=True)


# read fundamental statistics
fundamentals = pd.DataFrame()
fundamentals['symbol'] = ratios['symbol']
fundamentals['reporting_date'] = ratios['date']
fundamentals['return_on_equity'] = ratios['Return on Equity / Networth (%)']
fundamentals['current_ratio'] = ratios['Current Ratio (X)']
fundamentals['quick_ratio'] = ratios['Quick Ratio (X)']
fundamentals['debt_to_equity'] = ratios['Total Debt/Equity (X)']
fundamentals['earnings_per_share'] = ratios['Diluted EPS (Rs.)']
fundamentals['bookvalue'] = ratios['Book Value [Excl. Reval Reserve]/Share (Rs.)']
fundamentals['adj_eps'] = ratios['adj_eps']
fundamentals['adj_bookvalue'] = ratios['adj_bookvalue']
fundamentals['eps_growth'] = ratios['eps_growth']


# prepare master data
chunks = np.array_split(prices, 10)
master_data = []
for chunk in chunks:
    temp = chunk.merge(fundamentals, how='left', left_on='symbol', right_on='symbol')
    temp = temp[(temp['Date'] > temp['reporting_date']) & (temp['reporting_date'] > temp['Date'] + pd.Timedelta(-380, unit='d'))]
    master_data.append(temp)
master_data = pd.concat(master_data, ignore_index=True)
master_data['price_to_earnings'] = master_data['adj_close']/master_data['adj_eps']
master_data['price_to_bookvalue'] = master_data['adj_close']/master_data['adj_bookvalue']

del temp, chunks
master_data = master_data.merge(stockdetail, left_on='symbol', right_on='Symbol')

# prepare market index
mktidx.sort_values('Date', inplace=True)
mktidx['benchmark_returns'] = 100*mktidx['Close'].pct_change(return_window).shift(-return_window).values
mktidx = mktidx[['Date', 'benchmark_returns']]
mktidx.dropna(inplace=True)
master_data = master_data.merge(mktidx, how='left', on='Date')

master_data['year'] = master_data.Date.apply(lambda x: x.year)
master_data['month'] = master_data.Date.apply(lambda x: x.month)




# prepare training data
train_idx = (master_data.Date > '2000-12-31') #& (master_data.Date <= '2015-12-31')

X = master_data.drop(['symbol', 'Symbol',
                       'returns', 'benchmark_returns', 'daily_returns',
                       'Date' , 'reporting_date', 'year', 'month',
                       'Open', 'Close', 'Low', 'High',
                       'adj_open', 'adj_close', 'adj_low', 'adj_high', 
                       'Last', 'TotalTradeQuantity', 'TurnoverLacs',
                       'earnings_per_share', 'bookvalue',
                       'adj_eps', 'adj_bookvalue',
                       'Industry'], axis=1)
X['target'] = master_data['returns'] > master_data['benchmark_returns'] + 10
X['target'] = X['target'].apply(int)
#X['Industry'] = X['Industry'].map(X.groupby('Industry')['target'].mean().to_dict())
#X['year'] = X['year'].map(X.groupby('year')['target'].mean().to_dict())
#X['month'] = X['month'].map(X.groupby('month')['target'].mean().to_dict())
y = X['target']
X.drop('target', axis=1, inplace=True)


# train
X_train, y_train, X_test, y_test = X[train_idx], y[train_idx], X[~train_idx], y[~train_idx]
X_train.fillna(X_train.min() - 2*(X_train.max()-X_train.min()), inplace=True)
X_train.replace([np.inf, -np.inf], 100000, inplace=True)
if len(X_test) > 0:
    X_test.fillna(X_train.min() - 2*(X_train.max()-X_train.min()), inplace=True)
    X_test.replace([np.inf, -np.inf], 100000, inplace=True)
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
clf.fit(X_train, y_train)
RF_importance = pd.DataFrame({'importance': sorted(clf.feature_importances_)}, index=X_train.columns[clf.feature_importances_.argsort()])
if len(X_test) > 0:
    preds = clf.predict_proba(X_test)[:, 1]
    print(np.sum((y_test.values == 0) & (preds > 0.9))/np.sum(preds > 0.9))

joblib.dump(clf, 'rf.joblib')
joblib.dump(X_train, 'train_data.joblib')
