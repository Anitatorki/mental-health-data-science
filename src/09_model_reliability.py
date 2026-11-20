"""
Day 11: Overfitting, Underfitting, and Model Reliability

Compare a baseline, linear regression, and decision trees with different
depths using training and held-out test performance.
Educational example using synthetic data; not a clinical tool.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mental_health_synthetic.csv"
FEATURES = ["sleep_hours", "exercise_days_per_week", "work_study_hours"]
TARGET = "wellbeing_score"


def rmse(actual, predicted):
    return mean_squared_error(actual, predicted) ** 0.5


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "Mean baseline": make_pipeline(
            SimpleImputer(strategy="median"),
            DummyRegressor(strategy="mean"),
        ),
        "Linear regression": make_pipeline(
            SimpleImputer(strategy="median"),
            LinearRegression(),
        ),
        "Decision tree (depth 1)": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=1, random_state=42),
        ),
        "Decision tree (depth 3)": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=3, random_state=42),
        ),
        "Decision tree (unrestricted)": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(random_state=42),
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_rmse = rmse(y_train, train_predictions)
        test_rmse = rmse(y_test, test_predictions)

        results.append(
            {
                "Model": name,
                "Train MAE": mean_absolute_error(y_train, train_predictions),
                "Test MAE": mean_absolute_error(y_test, test_predictions),
                "Train RMSE": train_rmse,
                "Test RMSE": test_rmse,
                "RMSE gap": test_rmse - train_rmse,
            }
        )

    results_df = pd.DataFrame(results).sort_values("Test RMSE")
    print("Model reliability comparison (lower MAE/RMSE is better):")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    print(
        "\\nInterpretation guide:\\n"
        "- High training and test error: possible underfitting.\\n"
        "- Very low training error but much higher test error: possible overfitting.\\n"
        "- Compare test error with the mean baseline before preferring a complex model.\\n"
        "This small synthetic dataset is for practice only."
    )


if __name__ == "__main__":
    main()
