import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
prior = pd.read_csv("data/raw/order_products__prior.csv")
products = pd.read_csv("data/raw/products.csv")

print("orders:", orders.shape)
print("prior:", prior.shape)
print("products:", products.shape)
print(orders.head())