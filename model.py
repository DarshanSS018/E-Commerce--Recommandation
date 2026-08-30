"""
E-commerce recommendation models: preprocessing, collaborative filtering,
content-based filtering, evaluation helpers, and trending/search utilities.

This module is written to be readable for beginners: each step is commented.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# -----------------------------------------------------------------------------
# Paths (same folder as this file; data lives in ./data/)
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


def load_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the three core tables: category hierarchy, products, and ratings.

    Returns:
        category_tree: categoryid, parentid
        products: product metadata (names, categories, text for content-based)
        ratings: user_id, product_id, rating, timestamp
    """
    cat_path = os.path.join(data_dir, "category_tree.csv")
    prod_path = os.path.join(data_dir, "products.csv")
    rat_path = os.path.join(data_dir, "ratings.csv")

    category_tree = pd.read_csv(cat_path)
    products = pd.read_csv(prod_path)
    ratings = pd.read_csv(rat_path)
    return category_tree, products, ratings


def preprocess_data(
    category_tree: pd.DataFrame,
    products: pd.DataFrame,
    ratings: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Preprocessing pipeline:
    - Missing values: fill empty parent IDs, drop bad rating rows
    - Categorical -> numeric: one-hot encode product category (for analysis / optional blending)
    - Normalize: scale price to [0, 1] for content features

    Returns a dict with cleaned frames and fitted scaler / encoder (for consistency in the app).
    """
    # --- Category tree: parentid may be missing for root-like nodes ---
    ct = category_tree.copy()
    ct["parentid"] = pd.to_numeric(ct["parentid"], errors="coerce")
    ct["parentid"] = ct["parentid"].fillna(-1).astype(int)

    # --- Products: drop rows without category or name ---
    pr = products.copy()
    pr = pr.dropna(subset=["category_id", "product_name"])
    pr["category_id"] = pr["category_id"].astype(int)
    pr["price"] = pd.to_numeric(pr["price"], errors="coerce")
    pr["price"] = pr["price"].fillna(pr["price"].median())

    # --- Ratings: valid users, products, 1-5 stars ---
    rt = ratings.copy()
    rt = rt.dropna(subset=["user_id", "product_id", "rating"])
    rt["user_id"] = rt["user_id"].astype(int)
    rt["product_id"] = rt["product_id"].astype(int)
    rt["rating"] = pd.to_numeric(rt["rating"], errors="coerce")
    rt = rt.dropna(subset=["rating"])
    rt = rt[(rt["rating"] >= 1) & (rt["rating"] <= 5)]

    # Keep only ratings for products that exist after product cleaning
    valid_pids = set(pr["product_id"].unique())
    rt = rt[rt["product_id"].isin(valid_pids)]

    # --- Encode category as numeric columns (one-hot) for optional use ---
    try:
        ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
    except TypeError:  # scikit-learn < 1.2
        ohe = OneHotEncoder(sparse=True, handle_unknown="ignore")  # type: ignore[call-arg]
    cat_matrix = ohe.fit_transform(pr[["category_id"]])

    # --- Normalize price (useful as a side feature in content models) ---
    price_scaler = MinMaxScaler()
    pr["price_normalized"] = price_scaler.fit_transform(pr[["price"]])

    return {
        "category_tree": ct,
        "products": pr.reset_index(drop=True),
        "ratings": rt.reset_index(drop=True),
        "category_onehot": cat_matrix,
        "category_ohe": ohe,
        "price_scaler": price_scaler,
    }


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Wide matrix: rows = users, columns = products, values = star rating (NaN = no rating)."""
    return ratings.pivot_table(
        index="user_id",
        columns="product_id",
        values="rating",
        aggfunc="mean",
    )


def _pairwise_user_cosine(rating_matrix: np.ndarray) -> np.ndarray:
    """
    User–user cosine similarity (vectorized).
    Each row is mean-centered (NaN ignored for the mean), missing entries are 0,
    rows are L2-normalized, then sim = X @ X.T. Diagonal is set to 0.
    """
    r = rating_matrix.astype(np.float64, copy=False)
    user_means = np.nanmean(r, axis=1, keepdims=True)
    user_means = np.nan_to_num(user_means, nan=0.0)
    centered = np.where(np.isnan(r), 0.0, r - user_means)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x = centered / norms
    sim = x @ x.T
    np.fill_diagonal(sim, 0.0)
    return sim.astype(np.float64)


def _user_index_map(user_ids: np.ndarray) -> Dict[int, int]:
    return {int(u): idx for idx, u in enumerate(user_ids)}


class UserBasedCollaborativeFiltering:
    """
    User-based collaborative filtering:
    Find users with similar taste (cosine on co-rated items), then score
    unseen items with a weighted blend of neighbors' centered ratings.
    """

    def __init__(self, k_neighbors: int = 40):
        self.k_neighbors = k_neighbors
        self.user_ids_: Optional[np.ndarray] = None
        self.product_ids_: Optional[np.ndarray] = None
        self.rating_matrix_: Optional[np.ndarray] = None
        self.user_means_: Optional[np.ndarray] = None
        self.similarity_: Optional[np.ndarray] = None
        self.uid_to_row_: Dict[int, int] = {}
        self.pid_to_col_: Dict[int, int] = {}

    def fit(self, ratings_train: pd.DataFrame) -> "UserBasedCollaborativeFiltering":
        wide = build_user_item_matrix(ratings_train)
        self.user_ids_ = wide.index.values.astype(int)
        self.product_ids_ = wide.columns.values.astype(int)
        self.rating_matrix_ = wide.values.astype(float)
        # Per-user mean (ignore NaN) — used to center predictions
        self.user_means_ = np.nanmean(self.rating_matrix_, axis=1)
        self.similarity_ = _pairwise_user_cosine(self.rating_matrix_)
        self.uid_to_row_ = _user_index_map(self.user_ids_)
        self.pid_to_col_ = {int(p): j for j, p in enumerate(self.product_ids_)}
        return self

    def predict_score(self, user_id: int, product_id: int) -> float:
        """Predict a single rating; returns global mean if user/item unknown."""
        if self.rating_matrix_ is None:
            raise RuntimeError("Call fit() first.")
        if user_id not in self.uid_to_row_ or product_id not in self.pid_to_col_:
            return float(np.nanmean(self.rating_matrix_))
        u = self.uid_to_row_[user_id]
        i = self.pid_to_col_[product_id]
        r_ui = self.rating_matrix_[u, i]
        if not np.isnan(r_ui):
            return float(r_ui)
        sim_row = self.similarity_[u].copy()
        neighbor_ratings = self.rating_matrix_[:, i]
        mask_neighbors = ~np.isnan(neighbor_ratings) & (np.arange(len(sim_row)) != u)
        if not mask_neighbors.any():
            return float(self.user_means_[u])
        sims = sim_row[mask_neighbors]
        ratings_n = neighbor_ratings[mask_neighbors]
        means_n = self.user_means_[mask_neighbors]
        centered = ratings_n - means_n
        top_idx = np.argsort(sims)[::-1][: self.k_neighbors]
        sims_k = sims[top_idx]
        cent_k = centered[top_idx]
        pos = sims_k > 0
        if not pos.any():
            return float(self.user_means_[u])
        w = sims_k[pos]
        num = (w * cent_k[pos]).sum()
        den = np.abs(w).sum() + 1e-9
        pred = self.user_means_[u] + num / den
        return float(np.clip(pred, 1.0, 5.0))

    def recommend(
        self,
        user_id: int,
        products: pd.DataFrame,
        n: int = 5,
        category_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Top-n products not yet rated by user; score = predicted rating."""
        if user_id not in self.uid_to_row_:
            return pd.DataFrame(columns=["product_id", "product_name", "category_id", "score"])
        u = self.uid_to_row_[user_id]
        row = self.rating_matrix_[u]
        pid_to_cat = dict(zip(products["product_id"].astype(int), products["category_id"].astype(int)))
        candidates: List[Tuple[int, float]] = []
        for j, pid in enumerate(self.product_ids_):
            if not np.isnan(row[j]):
                continue
            if category_ids is not None:
                cid = pid_to_cat.get(int(pid))
                if cid is None or int(cid) not in category_ids:
                    continue
            sc = self.predict_score(user_id, int(pid))
            candidates.append((int(pid), sc))
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:n]
        if not top:
            return pd.DataFrame(columns=["product_id", "product_name", "category_id", "score"])
        out = pd.DataFrame(top, columns=["product_id", "score"])
        out = out.merge(products[["product_id", "product_name", "category_id"]], on="product_id", how="left")
        return out


class ContentBasedRecommender:
    """
    Content-based filtering using TF-IDF on text + cosine similarity to a user profile.
    User profile = weighted average of TF-IDF vectors of items they rated (weighted by rating).
    """

    def __init__(self, max_features: int = 256):
        self.max_features = max_features
        self.vectorizer_: Optional[TfidfVectorizer] = None
        self.product_ids_: Optional[np.ndarray] = None
        self.tfidf_matrix_: Optional[csr_matrix] = None
        self._products_lookup: Optional[pd.DataFrame] = None

    def fit(self, products: pd.DataFrame) -> "ContentBasedRecommender":
        self._products_lookup = products.reset_index(drop=True)
        text = (
            self._products_lookup["product_name"].fillna("")
            + " "
            + self._products_lookup["description"].fillna("")
        )
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
        )
        tfidf = self.vectorizer_.fit_transform(text)
        price_col = csr_matrix(self._products_lookup[["price_normalized"]].values)
        self.tfidf_matrix_ = hstack([tfidf, price_col])
        self.product_ids_ = self._products_lookup["product_id"].values.astype(int)
        return self

    def _profile_vector(self, ratings_user: pd.DataFrame):
        """Weighted average TF-IDF (+ price) for one user. Returns None if no overlap."""
        if self.tfidf_matrix_ is None or self.product_ids_ is None:
            raise RuntimeError("Call fit() first.")
        pid_to_row = {int(pid): idx for idx, pid in enumerate(self.product_ids_)}
        ru = ratings_user.dropna(subset=["product_id", "rating"]).copy()
        ru["product_id"] = ru["product_id"].astype(int)
        valid = ru[ru["product_id"].isin(pid_to_row)]
        if valid.empty:
            return None
        row_ix = valid["product_id"].map(pid_to_row).to_numpy(dtype=np.intp)
        weights = valid["rating"].to_numpy(dtype=np.float64)
        w_norm = weights / (weights.sum() + 1e-9)
        batch = self.tfidf_matrix_[row_ix]
        dense = batch.toarray()
        profile = (dense * w_norm.reshape(-1, 1)).sum(axis=0, keepdims=True)
        return csr_matrix(profile)

    def recommend(
        self,
        user_id: int,
        ratings: pd.DataFrame,
        products: pd.DataFrame,
        n: int = 5,
        category_ids: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Recommend top-n products by cosine similarity between user profile and item vectors.
        `score` is cosine similarity in [0, 1] for comparable direction (higher = better match).
        """
        ratings_user = ratings[ratings["user_id"] == user_id]
        if ratings_user.empty:
            return pd.DataFrame(columns=["product_id", "product_name", "category_id", "score"])

        profile = self._profile_vector(ratings_user)
        if profile is None:
            return pd.DataFrame(columns=["product_id", "product_name", "category_id", "score"])

        seen = set(ratings_user["product_id"].astype(int).tolist())
        sims = cosine_similarity(profile, self.tfidf_matrix_)[0]

        pid_to_cat = dict(zip(products["product_id"].astype(int), products["category_id"].astype(int)))
        ranked: List[Tuple[int, float]] = []
        for pos, pid in enumerate(self.product_ids_):
            if int(pid) in seen:
                continue
            if category_ids is not None:
                cid = pid_to_cat.get(int(pid))
                if cid is None or int(cid) not in category_ids:
                    continue
            ranked.append((int(pid), float(sims[pos])))
        ranked.sort(key=lambda x: x[1], reverse=True)
        top = ranked[:n]
        if not top:
            return pd.DataFrame(columns=["product_id", "product_name", "category_id", "score"])
        out = pd.DataFrame(top, columns=["product_id", "score"])
        out = out.merge(products[["product_id", "product_name", "category_id"]], on="product_id", how="left")
        return out


# -----------------------------------------------------------------------------#
# Evaluation helpers
# -----------------------------------------------------------------------------#


def evaluate_collaborative_filtering(
    ratings: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    k_neighbors: int = 40,
) -> Dict[str, float]:
    """
    Hold out random (user, item) ratings, train user-based CF on the rest, report RMSE on test.
    RMSE is on the 1-5 star scale (lower is better).
    """
    idx = np.arange(len(ratings))
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state)
    train = ratings.iloc[train_idx]
    test = ratings.iloc[test_idx]
    model = UserBasedCollaborativeFiltering(k_neighbors=k_neighbors).fit(train)

    preds: List[float] = []
    actuals: List[float] = []
    for _, row in test.iterrows():
        u, p = int(row["user_id"]), int(row["product_id"])
        if u in model.uid_to_row_ and p in model.pid_to_col_:
            preds.append(model.predict_score(u, p))
            actuals.append(float(row["rating"]))
    if len(preds) < 5:
        return {"rmse": float("nan"), "n_test": int(len(preds))}
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    return {"rmse": rmse, "n_test": int(len(preds))}


def evaluate_content_similarity_for_user(
    user_id: int,
    ratings: pd.DataFrame,
    products: pd.DataFrame,
    cb_model: ContentBasedRecommender,
    top_k: int = 5,
) -> Dict[str, float]:
    """Average cosine score of the top-k content-based recommendations (higher = stronger match)."""
    rec = cb_model.recommend(user_id, ratings, products, n=top_k)
    if rec.empty:
        return {"mean_top_k_cosine": float("nan"), "top_k": int(top_k)}
    return {"mean_top_k_cosine": float(rec["score"].mean()), "top_k": int(top_k)}


# -----------------------------------------------------------------------------#
# Trending & search (extra features)
# -----------------------------------------------------------------------------#


def trending_products(ratings: pd.DataFrame, products: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Trending = high recent activity + average rating.
    Uses timestamp when present; falls back to volume * avg rating.
    """
    r = ratings.copy()
    if "timestamp" in r.columns:
        latest = r["timestamp"].max()
        r["recency"] = latest - r["timestamp"]
        half_life = r["recency"].median() + 1.0
        r["trend_weight"] = np.exp(-r["recency"] / (half_life + 1e-9))
    else:
        r["trend_weight"] = 1.0
    g = r.groupby("product_id").agg(avg_rating=("rating", "mean"), count=("rating", "count"), tw=("trend_weight", "sum"))
    g["trend_score"] = g["avg_rating"] * np.log1p(g["count"]) * (g["tw"] / g["count"].clip(lower=1))
    g = g.sort_values("trend_score", ascending=False).head(top_n).reset_index()
    return g.merge(products[["product_id", "product_name", "category_id"]], on="product_id", how="left")


def search_products(products: pd.DataFrame, query: str, limit: int = 20) -> pd.DataFrame:
    """Simple case-insensitive substring search on name + description."""
    if not query or not str(query).strip():
        return products.head(limit).copy()
    q = str(query).lower()
    mask = products["product_name"].fillna("").str.lower().str.contains(q, regex=False) | products[
        "description"
    ].fillna("").str.lower().str.contains(q, regex=False)
    return products.loc[mask].head(limit).reset_index(drop=True)


def all_category_options(products: pd.DataFrame) -> List[int]:
    return sorted(products["category_id"].dropna().astype(int).unique().tolist())