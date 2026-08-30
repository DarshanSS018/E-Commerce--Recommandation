"""
Streamlit web app: explore recommendations, charts, search, and trending products.

Run from the project root:
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import model

# Page setup
st.set_page_config(page_title="E-commerce Recommendations", layout="wide")
st.title("E-commerce product recommendations")
st.caption(
    "Demonstrates **user-based collaborative filtering** (predicted rating as score) "
    "and **content-based filtering** (cosine similarity to your taste profile)."
)


@st.cache_resource(show_spinner="Loading data and training models…")
def get_pipeline():
    """Load CSVs, preprocess, and fit both recommenders once per session."""
    cat, products_raw, ratings_raw = model.load_data(model.DATA_DIR)
    pack = model.preprocess_data(cat, products_raw, ratings_raw)
    products = pack["products"]
    ratings = pack["ratings"]
    cf = model.UserBasedCollaborativeFiltering(k_neighbors=40).fit(ratings)
    cb = model.ContentBasedRecommender(max_features=256).fit(products)
    cf_metrics = model.evaluate_collaborative_filtering(ratings, test_size=0.2, random_state=42)
    return {
        "category_tree": pack["category_tree"],
        "products": products,
        "ratings": ratings,
        "cf": cf,
        "cb": cb,
        "cf_metrics": cf_metrics,
    }


pipe = get_pipeline()
products: pd.DataFrame = pipe["products"]
ratings: pd.DataFrame = pipe["ratings"]
cf_model = pipe["cf"]
cb_model = pipe["cb"]

min_uid = int(ratings["user_id"].min())
max_uid = int(ratings["user_id"].max())
categories = model.all_category_options(products)

# --- Sidebar: controls ---
with st.sidebar:
    st.header("Your settings")
    user_id = int(
        st.number_input(
            "User ID",
            min_value=min_uid,
            max_value=max_uid,
            value=min_uid,
            step=1,
            help=f"Sample users range from {min_uid} to {max_uid}.",
        )
    )
    method = st.radio("Recommendation method", ["Collaborative filtering", "Content-based", "Compare both"])

    st.subheader("Optional filters")
    search_q = st.text_input("Search catalog (name / description)", "")
    filter_cats = st.multiselect(
        "Limit to category IDs (empty = all categories)",
        options=categories,
        default=[],
    )
    cat_filter = filter_cats if filter_cats else None

    st.subheader("Quick evaluation snapshot")
    st.metric("CF hold-out RMSE (stars)", f"{pipe['cf_metrics']['rmse']:.3f}")
    st.caption(
        f"Based on {int(pipe['cf_metrics']['n_test'])} held-out ratings. "
        "Lower RMSE means better rating prediction for collaborative filtering."
    )


# --- Main area ---
tab_rec, tab_viz, tab_trend = st.tabs(["Recommendations", "Insights & charts", "Trending & search"])

with tab_rec:
    if method == "Collaborative filtering":
        rec = cf_model.recommend(user_id, products, n=5, category_ids=cat_filter)
        rec = rec.rename(columns={"score": "predicted_rating"})
        st.subheader("Top 5 for you (collaborative)")
        st.dataframe(rec, use_container_width=True)
        if not rec.empty:
            st.caption("Score = predicted star rating (1–5) from similar users.")

    elif method == "Content-based":
        rec = cb_model.recommend(user_id, ratings, products, n=5, category_ids=cat_filter)
        rec = rec.rename(columns={"score": "cosine_similarity"})
        st.subheader("Top 5 for you (content-based)")
        st.dataframe(rec, use_container_width=True)
        if not rec.empty:
            st.caption("Score = cosine similarity to your profile (higher = closer match).")

    else:
        c1, c2 = st.columns(2)
        rec_cf = cf_model.recommend(user_id, products, n=5, category_ids=cat_filter)
        rec_cb = cb_model.recommend(user_id, ratings, products, n=5, category_ids=cat_filter)
        cb_diag = model.evaluate_content_similarity_for_user(user_id, ratings, products, cb_model, top_k=5)
        with c1:
            st.markdown("**Collaborative filtering** (predicted rating)")
            st.dataframe(rec_cf.rename(columns={"score": "predicted_rating"}), use_container_width=True)
        with c2:
            st.markdown("**Content-based** (cosine similarity)")
            st.dataframe(rec_cb.rename(columns={"score": "cosine_similarity"}), use_container_width=True)
        cos = cb_diag["mean_top_k_cosine"]
        cos_txt = f"{cos:.3f}" if cos == cos else "n/a"
        st.info(
            f"Held-out **RMSE** (collaborative): **{pipe['cf_metrics']['rmse']:.3f}** stars. "
            f"Mean top-5 **cosine** for this user (content): **{cos_txt}**."
        )

    st.divider()
    st.subheader("This user's rating history (sample)")
    hist = ratings[ratings["user_id"] == user_id].merge(
        products[["product_id", "product_name"]], on="product_id", how="left"
    )
    st.dataframe(hist.sort_values("rating", ascending=False).head(15), use_container_width=True)

with tab_viz:
    st.subheader("Dataset insights")

    pop = (
        ratings.groupby("product_id")
        .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .merge(products[["product_id", "product_name"]], on="product_id")
        .sort_values("rating_count", ascending=False)
        .head(15)
    )

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.barh(pop["product_name"].str.slice(0, 28)[::-1], pop["rating_count"][::-1], color="steelblue")
    ax1.set_xlabel("Number of ratings")
    ax1.set_title("Most popular products (by rating count)")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    act = ratings.groupby("user_id").size().reset_index(name="n_ratings")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.hist(act["n_ratings"], bins=20, color="darkorange", edgecolor="white")
    ax2.set_xlabel("Ratings per user")
    ax2.set_ylabel("Users")
    ax2.set_title("User activity distribution")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax3, color="seagreen")
    ax3.set_xlabel("Star rating")
    ax3.set_ylabel("Count")
    ax3.set_title("Ratings distribution")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

with tab_trend:
    st.subheader("Trending products")
    trend = model.trending_products(ratings, products, top_n=12)
    st.dataframe(trend, use_container_width=True)

    st.subheader("Catalog search results")
    if not str(search_q).strip():
        st.caption("Enter a search term above to filter the catalog.")
    else:
        hits = model.search_products(products, search_q, limit=25)
        st.dataframe(hits[["product_id", "product_name", "category_id", "price"]], use_container_width=True)
