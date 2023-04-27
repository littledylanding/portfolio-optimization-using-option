import pandas as pd
import numpy as np
from mvp import mvp

# Read data
df_map = pd.read_excel('stocks.xlsx', sheet_name=None)
data = pd.DataFrame([])
for k, v in df_map.items():
    temp = v
    temp.index = temp['Date']
    temp.drop(['Date'], axis=1, inplace=True)
    temp.columns = [k]
    data = pd.concat([data, temp], axis=1)
data.index = pd.to_datetime(data.index)
data.sort_index(inplace=True)

# Set up
rolling_period = 21

# Construct portfolio
portfolio = mvp(price=data, rolling_period=rolling_period, type='equal')

# Daily trading
for i in range(rolling_period, len(data)):
    portfolio.start()

ret = pd.DataFrame(portfolio.ret)
ret.index = data.index[rolling_period:]
temp = pd.DataFrame([portfolio.turnover])
temp.index = [ret.index[-1]+pd.Timedelta(days=1)]
ret = pd.concat([ret, temp])
ret.to_csv('equal.csv')
