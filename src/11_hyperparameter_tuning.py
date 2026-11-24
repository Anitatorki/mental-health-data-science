"""
Day 13: Hyperparameter tuning with GridSearchCV.

Tunes a decision tree using cross-validation on the training split,
then evaluates the selected pipeline once on a held-out test set.
Educational synthetic-data example; not a clinical tool.
"""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "mental_health_synthetic.csv"
FEATURES = ["sleep_hours", "exercise_days_per_week", "work_study_hours"]
TARGET = "wellbeing_score"


def evaluate(model, X, y):
    predictions = model.predict(X)
    rmse = mean_squared_error(y, predictions) ** 0.5
    mae = mean_absolute_error(y, predictions)
    return rmse, mae


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET]

    # Keep this test set out of hyperparameter selection.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]
    )

    parameter_grid = {
        "model__max_depth": [1, 2, 3, 4, 5, None],
        "model__min_samples_leaf": [1, 2, 4, 6],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)

    best_cv_rmse = -search.best_score_
    tuned_test_rmse, tuned_test_mae = evaluate(search.best_estimator_, X_test, y_test)

    baseline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_test_rmse, baseline_test_mae = evaluate(baseline, X_test, y_test)

    print("Best hyperparameters:")
    print(search.best_params_)
    print(f"Best mean CV RMSE: {best_cv_rmse:.3f}")

    print("\nHeld-out test comparison (lower is better):")
    print(f"Tuned tree  - RMSE: {tuned_test_rmse:.3f}, MAE: {tuned_test_mae:.3f}")
    print(f"Mean baseline - RMSE: {baseline_test_rmse:.3f}, MAE: {baseline_test_mae:.3f}")

    if tuned_test_rmse < baseline_test_rmse:
        print("\nThe tuned tree improved on the baseline for this test split.")
    else:
        print("\nThe tuned tree did not beat the baseline on this test split.")

    print(
        "Treat these results as a coding exercise only. "
        "The dataset is synthetic and not suitable for clinical conclusions."
    )


if __name__ == "__main__":
    main()
