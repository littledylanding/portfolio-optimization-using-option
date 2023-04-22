import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def calMVSharpe(retForCal):
    timeLength = (retForCal.index[-1] - retForCal.index[0]).days / 365.3
    yearRet = round(np.nanmean(retForCal) * 250, 4)
    stdRet = np.nanstd(retForCal)
    volatT = stdRet * np.sqrt(len(retForCal) / timeLength)
    shp = round(yearRet / volatT, 3)
    return shp


res = []
files = ['normal.csv', 'option_implied.csv']
for f in files:
    data = pd.read_csv(f, index_col=0)
    data.index = pd.to_datetime(data.index)
    turnover = data.iloc[-1]
    data = data.iloc[:-1]
    total_ret = np.sum(data)
    sharpe = calMVSharpe(data)
    res.append([total_ret, sharpe, turnover])
res = pd.DataFrame(res)
res.columns = ['total return', 'sharpe ratio', 'turnover']
res.index = ['normal', 'option implied']