import numpy as np
import pandas as pd

# Set up
rolling_period = 21

# Read data
df_map = pd.read_excel('output.xlsx', sheet_name=None)
RV = pd.DataFrame([])
for k, v in df_map.items():
    temp = v
    temp.index = temp['Date']
    rv = temp['predicted_RV']
    rv.rename(k, inplace=True)
    RV = pd.concat([RV, rv], axis=1)

RV.index = pd.to_datetime(RV.index)
RV.sort_index(inplace=True)
RV = RV.values
RV = np.concatenate([np.zeros((rolling_period-1, RV.shape[1])), RV])
RV = pd.DataFrame(RV)

df_map = pd.read_excel('origin.xlsx', sheet_name=None)
price = pd.DataFrame([])
for k, v in df_map.items():
    temp = v
    temp.index = temp['Date']
    temp.index = pd.to_datetime(temp.index)
    temp.sort_index(inplace=True)
    p = temp['Close'].iloc[len(temp)-len(RV):, ]
    p.rename(k, inplace=True)
    price = pd.concat([price, p], axis=1)
