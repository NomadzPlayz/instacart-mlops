import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("analysis/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_parquet("data/processed/features.parquet")

# 1. Label distribution
df["label"].value_counts().plot(kind="bar")
plt.title("Label Distribution (Reorder vs Non-Reorder)")
plt.xlabel("Label")
plt.ylabel("Count")
plt.savefig("analysis/figures/label_distribution.png", dpi=150)
plt.close()

# 2. User order count distribution
df["user_orders"].hist(bins=50)
plt.title("Distribution of User Order Counts")
plt.xlabel("Number of Orders")
plt.ylabel("Users")
plt.savefig("analysis/figures/user_orders_distribution.png", dpi=150)
plt.close()

# 3. Product popularity
df["product_orders"].hist(bins=50)
plt.title("Distribution of Product Orders")
plt.xlabel("Total Product Orders")
plt.ylabel("Products")
plt.savefig("analysis/figures/product_orders_distribution.png", dpi=150)
plt.close()

print("EDA figures saved to analysis/figures/")