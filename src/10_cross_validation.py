"""
Day 12: Cross-validation for model comparison.

Uses cross-validation on the training split, then evaluates the selected
model once on a held-out test set. Educational synthetic-data example,
not a clinical tool.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
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

    # Keep the final test set out of cross-validation and model selection.
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
        "Decision tree (depth 2)": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=2, random_state=42),
        ),
        "Decision tree (depth 4)": make_pipeline(
            SimpleImputer(strategy="median"),
            DecisionTreeRegressor(max_depth=4, random_state=42),
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, model in models.items():
        # sklearn's neg_root_mean_squared_error is negative because
        # cross_val_score follows the convention that larger is better.
        scores = -cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        )
        rows.append(
            {
                "Model": name,
                "Mean CV RMSE": scores.mean(),
                "CV RMSE std": scores.std(),
            }
        )

    results = pd.DataFrame(rows).sort_values("Mean CV RMSE")
    print("5-fold cross-validation (lower RMSE is better):")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))

    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]
    best_model.fit(X_train, y_train)
    test_rmse = rmse(y_test, best_model.predict(X_test))

    print(f"\nSelected by mean CV RMSE: {best_name}")
    print(f"Final held-out test RMSE: {test_rmse:.3f}")
    print(
        "Interpret the test score only after model selection. "
        "This small synthetic dataset is for practice, not clinical use."
    )


if __name__ == "__main__":
    main()
