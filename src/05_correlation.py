"""
Day 7 — Correlation in Mental Health & Wellbeing Data Science

Learning goals:
1. Understand what correlation measures.
2. Calculate Pearson correlations with pandas.
3. Interpret direction and strength cautiously.

This lesson uses fictional, synthetic data only. It contains no real
participant or patient information.
"""

import pandas as pd

data = {
    "participant_id": [1, 2, 3, 4, 5, 6],
    "stress_score": [12, 18, 15, 22, 10, 17],
    "sleep_hours": [8.0, 6.0, 7.0, 5.5, 8.5, 6.5],
    "wellbeing_score": [78, 60, 70, 52, 85, 66],
}

df = pd.DataFrame(data)

# Correlation ranges from -1 to +1.
# Positive: two variables tend to increase together.
# Negative: one tends to decrease as the other increases.
# Near 0: little or no linear relationship in this sample.
correlation_matrix = df[
    ["stress_score", "sleep_hours", "wellbeing_score"]
].corr(method="pearson")

print("Pearson correlation matrix:")
print(correlation_matrix.round(2))

sleep_wellbeing = df["sleep_hours"].corr(df["wellbeing_score"])
stress_wellbeing = df["stress_score"].corr(df["wellbeing_score"])

print(f"\nSleep hours vs. wellbeing: r = {sleep_wellbeing:.2f}")
print(f"Stress score vs. wellbeing: r = {stress_wellbeing:.2f}")

print(
    "\nInterpretation reminder: correlation describes an association, "
    "not cause and effect. This dataset is tiny and synthetic, so the "
    "results are for learning only—not clinical conclusions."
)
