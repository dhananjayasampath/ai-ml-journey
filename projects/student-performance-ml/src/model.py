import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/dataset.csv")

X = df[["Hours", "Sleep"]]
y = df["Score"]

model = LinearRegression()
model.fit(X, y)

print("Model trained successfully")