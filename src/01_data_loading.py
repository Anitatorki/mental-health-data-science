"""Day 3: Load and inspect the synthetic mental health dataset."""

import pandas as pd

DATA_PATH = "data/mental_health_synthetic.csv"

df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nFirst five rows:")
print(df.head())
