import pandas as pd

df = pd.read_csv(r'a3-p2.csv', encoding='utf-8')
headers = df.columns.tolist()

for i in range(4):
    print(headers[i] + " and " + headers[4] + ": " + str(df[headers[i]].corr(df[headers[4]])))