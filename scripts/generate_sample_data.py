"""
Regenerate data/products.csv and data/ratings.csv from data/category_tree.csv.

Run from project root:
    python scripts/generate_sample_data.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    np.random.seed(42)
    cats = pd.read_csv(os.path.join(DATA, "category_tree.csv"))
    cat_ids = cats["categoryid"].dropna().astype(int).unique()
    np.random.shuffle(cat_ids)
    sel_cats = cat_ids[:90]
    adjs = ["Pro", "Ultra", "Smart", "Eco", "Classic", "Mini", "Max", "Elite", "Basic", "Prime"]
    nouns = [
        "Widget",
        "Gadget",
        "Kit",
        "Bundle",
        "Set",
        "Device",
        "Tool",
        "Accessory",
        "Pack",
        "Unit",
        "Item",
        "Gear",
        "Essentials",
        "Collection",
        "Edition",
    ]
    rows = []
    for i in range(1, 101):
        cid = int(sel_cats[i % len(sel_cats)])
        name = f"{np.random.choice(adjs)} {np.random.choice(nouns)} {i}"
        desc = (
            f"Quality product in category {cid}. Durable design for everyday use. "
            "Popular choice among customers."
        )
        price = round(np.random.uniform(9.99, 299.99), 2)
        rows.append((i, name, cid, desc, price))
    products = pd.DataFrame(
        rows, columns=["product_id", "product_name", "category_id", "description", "price"]
    )
    products.to_csv(os.path.join(DATA, "products.csv"), index=False)

    users = 250
    prod_list = products["product_id"].values
    rrows = []
    for u in range(1, users + 1):
        n = np.random.randint(15, 61)
        prods = np.random.choice(prod_list, size=min(n, len(prod_list)), replace=False)
        for p in prods:
            rrows.append(
                (
                    u,
                    int(p),
                    int(np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.15, 0.35, 0.35])),
                )
            )
    ratings = pd.DataFrame(rrows, columns=["user_id", "product_id", "rating"])
    ratings = ratings.drop_duplicates(subset=["user_id", "product_id"])
    ts = pd.date_range("2024-01-01", periods=len(ratings), freq="h")
    ratings["timestamp"] = (ts.view("int64") // 10**9).astype(int)
    ratings.to_csv(os.path.join(DATA, "ratings.csv"), index=False)
    print("Wrote", os.path.join(DATA, "products.csv"), products.shape)
    print("Wrote", os.path.join(DATA, "ratings.csv"), ratings.shape)


if __name__ == "__main__":
    sys.path.insert(0, ROOT)
    main()
