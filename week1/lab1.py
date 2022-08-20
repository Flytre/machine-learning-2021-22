import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Iris.csv")

# Helpful things
# print(data.head(10))  # First X rows
# print(data.columns)  # Column names
# print(data.shape)  # Row ct, col ct
# print(data.mean()) #Mean column vals
# print(data["sepallength"].head(150)) #get column

data.plot()

plt.show()
