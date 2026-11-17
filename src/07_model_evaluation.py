"""
Day 9 — Evaluating a Linear Regression Model
Project: Mental Health & Wellbeing Data Science

Uses a small synthetic dataset for learning only.
This is not a clinical tool and must not be used for individual predictions.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = "data/mental_health_synthetic.csv"

# 1. Load the data and select the feature and target.
df = pd.read_csv(DATA_PATH)
model_data = df[["sleep_hours", "wellbeing_score"]].dropna()

X = model_data[["sleep_hours"]]
y = model_data["wellbeing_score"]

# 2. Split before training so evaluation uses unseen test observations.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 3. Train the model using only the training data.
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict wellbeing scores for the test data.
predictions = model.predict(X_test)

# 5. Calculate common regression evaluation metrics.
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("=== Day 9: Regression Model Evaluation ===")
print(f"Training observations: {len(X_train)}")
print(f"Test observations: {len(X_test)}")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared on test data: {r2:.2f}")

# 6. Compare actual and predicted values.
comparison = pd.DataFrame(
    {
        "actual_wellbeing": y_test.values,
        "predicted_wellbeing": predictions.round(2),
    }
)
print("\nActual vs. predicted values:")
print(comparison.to_string(index=False))

# Interpretation:
# MAE and RMSE are in wellbeing-score points; lower is better.
# R-squared can be negative on test data when predictions are poor.
# With very few synthetic observations, these metrics are unstable.
# They do not establish causation or clinical usefulness.
