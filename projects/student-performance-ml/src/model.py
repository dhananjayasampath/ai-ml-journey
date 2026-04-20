import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("../data/dataset.csv") # Load the dataset

X = df[["Hours", "Sleep"]]  # Select the features (inputs)
y = df["Score"]  # Select the target variable (output)

model = LinearRegression() # Initialize the algorithm
model.fit(X, y) # Train the model using the data

print("Model trained successfully")