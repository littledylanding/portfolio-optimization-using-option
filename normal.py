import pandas as pd
import numpy as np
from mvp import mvp

# Set up
rolling_period = 21

# Read data
df_map = pd.read_excel('origin.xlsx', sheet_name=None)
price = pd.DataFrame([])
for k, v in df_map.items():
    temp = v
    temp.index = temp['Date']
    temp.index = pd.to_datetime(temp.index)
    temp.sort_index(inplace=True)
    p = temp['Close'].iloc[len(temp)-271:, ]
    p.rename(k, inplace=True)
    price = pd.concat([price, p], axis=1)

# Construct portfolio
portfolio = mvp(price=price, rolling_period=rolling_period)

# Daily trading
portfolio.trade()

ret = pd.DataFrame(portfolio.ret)
ret.index = price.index[rolling_period:]
temp = pd.DataFrame([portfolio.turnover])
temp.index = [ret.index[-1]+pd.Timedelta(days=1)]
ret = pd.concat([ret, temp])
ret.to_csv('normal.csv')
