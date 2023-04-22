import pandas as pd
import numpy as np
from mvp import mvp

# Read data
data = pd.read_csv

# Set up
rolling_period = 250

# Construct portfolio
portfolio = mvp(data)

# Daily trading
for i in range(rolling_period, len(data)):
    portfolio.start()

ret = pd.DataFrame(portfolio.ret)
ret.index = data.index[rolling_period:]
temp = pd.DataFrame([portfolio.turnover])
ret = pd.concat([ret, temp])
ret.to_csv('normal.csv')
