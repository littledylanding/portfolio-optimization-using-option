import pandas as pd

# Read data
df_map = pd.read_excel('stocks.xlsx', sheet_name=None)
res = pd.DataFrame([])
for k, v in df_map.items():
    print(v.columns)
    temp = v
    temp.index = temp['Date']
    temp.drop(['Date'], axis=1, inplace=True)
    temp.columns = [k]
    res = pd.concat([res, temp], axis=1)
res.index = pd.to_datetime(res.index)
