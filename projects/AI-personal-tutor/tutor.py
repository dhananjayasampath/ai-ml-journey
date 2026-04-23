import joblib

# Load trained model
model = joblib.load("student_model.pkl")

# User input
hours = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))

# Prediction
result = model.predict([[hours, sleep]])

# Output
if result[0] == 1:
    print("You will PASS")
else:
    print("You may FAIL")