"""
Day 6 — Data Visualization
Project: Mental Health & Wellbeing Data Science

This lesson uses a small synthetic dataset for practice.
It does not contain real participant or patient information.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Example data: each row represents a fictional participant.
data = {
    "participant_id": [1, 2, 3, 4, 5, 6],
    "stress_score": [12, 18, 15, 22, 10, 17],
    "sleep_hours": [8.0, 6.0, 7.0, 5.5, 8.5, 6.5],
    "wellbeing_score": [78, 60, 70, 52, 85, 66],
}

df = pd.DataFrame(data)

# Chart 1: Compare stress scores across fictional participants.
plt.figure(figsize=(8, 5))
plt.bar(df["participant_id"], df["stress_score"])
plt.title("Stress Scores by Participant")
plt.xlabel("Participant ID")
plt.ylabel("Stress Score")
plt.tight_layout()
plt.savefig("stress_scores.png")
plt.show()

# Chart 2: Explore the relationship between sleep and wellbeing.
plt.figure(figsize=(8, 5))
plt.scatter(df["sleep_hours"], df["wellbeing_score"])
plt.title("Sleep Hours and Wellbeing Score")
plt.xlabel("Sleep Hours")
plt.ylabel("Wellbeing Score")
plt.tight_layout()
plt.savefig("sleep_vs_wellbeing.png")
plt.show()

# Reminder: a pattern in a chart does not prove that one variable causes another.
print(df[["stress_score", "sleep_hours", "wellbeing_score"]].describe())
