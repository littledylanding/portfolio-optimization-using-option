import pandas as pd
import numpy as np
from mvp import mvp

# Read data
data = pd.read_csv

# Set up
rolling_period = 250

# Construct portfolio
portfolio = mvp()

# Daily trading
for i in range(rolling_period, len(data)):
    # Caculate covariance
    cov =
    # Trading
    portfolio.start(cov)

ret = portfolio.ret.append(portfolio.turnover)
ret = pd.DataFrame(ret)
ret.to_csv('option_implied.csv')
