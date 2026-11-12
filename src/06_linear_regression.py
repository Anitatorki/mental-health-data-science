"""
Day 8 — Introduction to Linear Regression
Project: Mental Health & Wellbeing Data Science

This lesson uses synthetic learning data only. It is not a clinical tool.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_PATH = "data/mental_health_synthetic.csv"

# 1. Load the dataset.
df = pd.read_csv(DATA_PATH)

# 2. Keep the two columns needed for this example.
# Drop rows where either value is missing.
model_data = df[["sleep_hours", "wellbeing_score"]].dropna()

# 3. Separate the input feature (X) and target (y).
X = model_data[["sleep_hours"]]
y = model_data["wellbeing_score"]

# 4. Create and train a simple linear regression model.
model = LinearRegression()
model.fit(X, y)

# 5. Display the fitted equation: wellbeing = intercept + coefficient * sleep.
print("=== Linear Regression: Sleep and Wellbeing ===")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Sleep coefficient: {model.coef_[0]:.2f}")
print(f"R-squared on this small dataset: {model.score(X, y):.2f}")

# 6. Make one example prediction for 7 hours of sleep.
example = pd.DataFrame({"sleep_hours": [7.0]})
predicted_wellbeing = model.predict(example)[0]
print(f"Predicted wellbeing score for 7 hours of sleep: {predicted_wellbeing:.2f}")

# Important:
# This is a small synthetic dataset. The model shows an association,
# not proof that sleep causes a change in wellbeing. It is not suitable
# for diagnosis, treatment, or real-world individual predictions.
