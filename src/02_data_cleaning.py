"""Day 4: Clean and validate the synthetic mental health dataset."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/mental_health_synthetic.csv")
OUTPUT_PATH = Path("data/processed/mental_health_cleaned.csv")

df = pd.read_csv(DATA_PATH)

print("Missing values before cleaning:")
print(df.isna().sum())

# Fill the missing sleep value with the median sleep duration.
sleep_median = df["sleep_hours"].median()
df["sleep_hours"] = df["sleep_hours"].fillna(sleep_median)

# Basic range validation for variables with known boundaries.
assert df["age"].between(18, 100).all()
assert df["sleep_hours"].between(0, 24).all()
assert df["stress_score"].between(0, 10).all()
assert df["wellbeing_score"].between(0, 100).all()
assert df["exercise_days_per_week"].between(0, 7).all()
assert df["work_study_hours"].between(0, 168).all()

print("\nMedian sleep hours used:", sleep_median)
print("\nMissing values after cleaning:")
print(df.isna().sum())

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\nCleaned dataset saved to:", OUTPUT_PATH)
