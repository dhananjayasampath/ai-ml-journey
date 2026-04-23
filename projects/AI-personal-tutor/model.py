import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib       # used to save and load machine learning models

# Sample dataset
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Sleep": [8, 7, 6, 6, 5, 5, 4, 4],
    "Result": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[["Hours", "Sleep"]]
y = df["Result"]

model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "student_model.pkl")

print("Model trained and saved")