"""Day 5: Exploratory Data Analysis (EDA)."""

import pandas as pd

DATA_PATH = "data/mental_health_synthetic.csv"

df = pd.read_csv(DATA_PATH)

print("=== Descriptive Statistics ===")
print(df[["age", "stress_score", "sleep_hours", "wellbeing_score"]].describe())

print("\n=== Mean Scores ===")
print("Average stress:", df["stress_score"].mean())
print("Average sleep:", df["sleep_hours"].mean())
print("Average wellbeing:", df["wellbeing_score"].mean())

print("\n=== Correlation Matrix ===")
print(df[["stress_score", "sleep_hours", "wellbeing_score"]].corr())

# EDA questions:
# 1. What is the average stress level?
# 2. What is the average sleep duration?
# 3. What is the average wellbeing score?
# 4. Does stress appear to be related to wellbeing?
# 5. Does sleep appear to be related to wellbeing?
