import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading raw data...")
    orders = pd.read_csv(RAW / "orders.csv")
    prior = pd.read_csv(RAW / "order_products__prior.csv")
    products = pd.read_csv(RAW / "products.csv")

    print("Merging tables...")
    df = prior.merge(orders, on="order_id", how="left")
    df = df.merge(products, on="product_id", how="left")

    print("Saving to parquet...")
    df.to_parquet(OUT / "base.parquet", index=False)

    print("Done. Shape:", df.shape)

if __name__ == "__main__":
    main()
