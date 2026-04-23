import joblib

# Load trained model
model = joblib.load("student_model.pkl")

# User input
hours = float(input("Enter study hours: "))
sleep = float(input("Enter sleep hours: "))

# Prediction
result = model.predict([[hours, sleep]])

# Probability
prob = model.predict_proba([[hours, sleep]])

pass_prob = prob[0][1] * 100
fail_prob = prob[0][0] * 100

# Output
if result[0] == 1:
    print(f"You will PASS (Probability: {pass_prob:.2f}%)")
else:
    print(f"You may FAIL (Probability: {fail_prob:.2f}%)")