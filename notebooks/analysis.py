"""
EDA script for the E-Commerce Product Recommendation System.

Run:
    python notebooks/analysis.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

products = pd.read_csv(ROOT / "data" / "products.csv")
interactions = pd.read_csv(ROOT / "data" / "interactions.csv", parse_dates=["timestamp"])

print("=== DATASET SUMMARY ===")
print(f"Products: {len(products)}")
print(f"Users: {interactions['user_id'].nunique()}")
print(f"Interactions: {len(interactions)}")

print("\n=== EVENT COUNTS ===")
print(interactions["event"].value_counts())

print("\n=== TOP CATEGORIES ===")
print(products["category"].value_counts())

popular = (
    interactions.groupby("product_id")["weight"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
    .merge(products[["product_id", "product_name"]], on="product_id")
)

plt.figure(figsize=(10, 5))
plt.bar(popular["product_name"], popular["weight"])
plt.xticks(rotation=45, ha="right")
plt.title("Top 10 Products by Weighted Interactions")
plt.ylabel("Interaction Weight")
plt.tight_layout()
plt.show()
