"""
Plant Species Classifier — Data Exploration
Quick summary stats and a text-based feature overview.
"""

import pandas as pd
from sklearn.datasets import load_iris


def explore() -> None:
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    species_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
    df["species"] = pd.Series(iris.target).map(species_map)

    print("=" * 55)
    print("  PLANT SPECIES DATASET — OVERVIEW")
    print("=" * 55)
    print(f"\nShape      : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Species    : {df['species'].unique().tolist()}")
    print(f"Class dist.:\n{df['species'].value_counts().to_string()}")

    print("\n── Summary Statistics ──────────────────────────────────")
    print(df.describe().round(2).to_string())

    print("\n── Mean per Species ────────────────────────────────────")
    print(df.groupby("species").mean().round(2).to_string())

    print("\n── Missing Values ──────────────────────────────────────")
    missing = df.isnull().sum()
    print(missing.to_string() if missing.any() else "None — dataset is clean.")


if __name__ == "__main__":
    explore()