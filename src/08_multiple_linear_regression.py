"""
Day 10: Multiple Linear Regression

Predict a synthetic wellbeing score from several lifestyle features.
This is an educational example, not a clinical or diagnostic tool.
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mental_health_synthetic.csv"

# 1. Load the synthetic dataset.
df = pd.read_csv(DATA_PATH)

# 2. Select multiple predictors (X) and one outcome (y).
features = ["sleep_hours", "exercise_days_per_week", "work_study_hours"]
target = "wellbeing_score"

X = df[features]
y = df[target]

# 3. Split before fitting so the test data remains unseen.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# 4. Impute missing predictor values using training data, then fit regression.
model = make_pipeline(
    SimpleImputer(strategy="median"),
    LinearRegression()
)
model.fit(X_train, y_train)

# 5. Predict and evaluate on the held-out test set.
predictions = model.predict(X_test)

print("Multiple Linear Regression — Wellbeing Score")
print(f"Features: {features}")
print(f"MAE:  {mean_absolute_error(y_test, predictions):.2f}")
print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.2f}")
print(f"R-squared: {r2_score(y_test, predictions):.2f}")

# 6. Display learned coefficients (association within this fitted model).
regression = model.named_steps["linearregression"]
print("\nModel coefficients:")
for feature, coefficient in zip(features, regression.coef_):
    print(f"{feature}: {coefficient:.3f}")
print(f"Intercept: {regression.intercept_:.3f}")

print(
    "\nReminder: this dataset is small and synthetic. "
    "Results are for learning only and do not establish causation."
)
