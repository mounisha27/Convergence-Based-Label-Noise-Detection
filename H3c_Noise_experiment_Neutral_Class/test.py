import pandas as pd
df = pd.read_csv('Data.csv')
print(len(df))
print(df.columns.tolist())