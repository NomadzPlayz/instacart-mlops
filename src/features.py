import pandas as pd
from pathlib import Path

DATA = Path("data/processed/base.parquet")
OUT = Path("data/processed/features.parquet")
OUT.parent.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading base parquet...")
    df = pd.read_parquet(DATA)

    df = df.sort_values(["user_id", "order_number"])

    df["label"] = (
        df.groupby(["user_id", "product_id"])["order_number"]
        .shift(-1)
        .notnull()
        .astype(int)
    )

    print("Building features...")

    user_feats = (
        df.groupby("user_id")
        .agg(
            user_orders=("order_id", "nunique"),
            avg_days_between=("days_since_prior_order", "mean"),
        )
        .reset_index()
    )

    product_feats = (
        df.groupby("product_id")
        .agg(
            product_orders=("order_id", "nunique"),
        )
        .reset_index()
    )

    up_feats = (
        df.groupby(["user_id", "product_id"])
        .agg(
            label=("label", "max"),
        )
        .reset_index()
    )

    feats = (
        up_feats
        .merge(user_feats, on="user_id", how="left")
        .merge(product_feats, on="product_id", how="left")
    )

    feats.to_parquet(OUT, index=False)
    print("Done:", feats.shape)

if __name__ == "__main__":
    main()
