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

# ---------------------------------------------------------------------------
# Page configuration — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — minimal, professional data-science dashboard theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---------- global typography ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- page background ---------- */
    .stApp {
        background: #0f1117;
    }

    /* ---------- sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #1a1d27;
        border-right: 1px solid #2a2d3e;
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #a78bfa;
    }

    /* ---------- top header ---------- */
    .dash-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e3a5f 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid #4338ca;
    }
    .dash-header h1 {
        color: #e0e7ff;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0 0 4px 0;
    }
    .dash-header p {
        color: #a5b4fc;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ---------- section headings ---------- */
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #c4b5fd;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 12px;
        border-left: 3px solid #7c3aed;
        padding-left: 10px;
    }

    /* ---------- metric cards ---------- */
    [data-testid="metric-container"] {
        background: #1e2130;
        border: 1px solid #2e3250;
        border-radius: 10px;
        padding: 8px 16px;
    }
    [data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.75rem !important;
    }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #a78bfa !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* ---------- dataframe / table ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2e3250;
    }

    /* ---------- tab styling ---------- */
    .stTabs [data-baseweb="tab-list"] {
        background: #1a1d27;
        border-radius: 8px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
        border-radius: 6px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background: #312e81 !important;
        color: #e0e7ff !important;
    }

    /* ---------- info / success boxes ---------- */
    .stAlert {
        border-radius: 8px;
    }

    /* ---------- score badge ---------- */
    .score-high   { color: #4ade80; font-weight: 600; }
    .score-medium { color: #facc15; font-weight: 600; }
    .score-low    { color: #f87171; font-weight: 600; }

    /* ---------- divider ---------- */
    hr { border-color: #2e3250; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load data & train models (cached once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⏳ Loading data and training models…")
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

# ---------------------------------------------------------------------------
# Helper — add price column to recommendation result and format score
# ---------------------------------------------------------------------------

def enrich_recommendations(rec: pd.DataFrame, score_label: str) -> pd.DataFrame:
    """Merge price into the recommendation frame and rename the score column."""
    if rec.empty:
        return rec
    rec = rec.merge(products[["product_id", "price"]], on="product_id", how="left")
    rec = rec.rename(columns={"score": score_label})
    # Reorder columns for clean display
    cols = ["product_id", "product_name", "category_id", "price", score_label]
    return rec[[c for c in cols if c in rec.columns]]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛍️ Recommendation Settings")
    st.markdown("---")

    user_id = int(
        st.number_input(
            "👤 User ID",
            min_value=min_uid,
            max_value=max_uid,
            value=min_uid,
            step=1,
            help=f"Sample users range from {min_uid} to {max_uid}.",
        )
    )

    method = st.radio(
        "🔧 Recommendation Method",
        ["Collaborative Filtering", "Content-Based", "Compare Both"],
        help=(
            "**Collaborative Filtering** — finds similar users and predicts ratings.\n\n"
            "**Content-Based** — matches items to your taste profile using TF-IDF + cosine similarity.\n\n"
            "**Compare Both** — shows results side-by-side."
        ),
    )

    st.markdown("---")
    st.markdown("### 🔍 Optional Filters")

    search_q = st.text_input("Search catalog (name / description)", "")
    filter_cats = st.multiselect(
        "Limit to category IDs (empty = all)",
        options=categories,
        default=[],
    )
    cat_filter = filter_cats if filter_cats else None

    st.markdown("---")
    st.markdown("### 📊 Model Evaluation")
    rmse_val = pipe["cf_metrics"]["rmse"]
    n_test = int(pipe["cf_metrics"]["n_test"])
    st.metric(
        label="CF Hold-out RMSE (stars)",
        value=f"{rmse_val:.3f}" if rmse_val == rmse_val else "n/a",
    )
    st.caption(
        f"Evaluated on {n_test:,} held-out ratings. "
        "Lower RMSE = better collaborative-filtering prediction."
    )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="dash-header">
        <h1>🛍️ E-Commerce Recommendation System</h1>
        <p>
            User-based <strong>Collaborative Filtering</strong> (predicted rating score) &nbsp;·&nbsp;
            <strong>Content-Based Filtering</strong> (TF-IDF + cosine similarity) &nbsp;·&nbsp;
            Built with scikit-learn &amp; Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_rec, tab_viz, tab_trend = st.tabs(
    ["🎯  Recommendations", "📈  Insights & Charts", "🔥  Trending & Search"]
)

# ── Tab 1 · Recommendations ────────────────────────────────────────────────
with tab_rec:

    if method == "Collaborative Filtering":
        rec = cf_model.recommend(user_id, products, n=5, category_ids=cat_filter)
        rec = enrich_recommendations(rec, "predicted_rating")

        st.markdown('<p class="section-title">Top 5 Recommendations — Collaborative Filtering</p>', unsafe_allow_html=True)
        if rec.empty:
            st.warning("No recommendations found for this user with the current filters.")
        else:
            st.dataframe(
                rec.style.format({"price": "${:.2f}", "predicted_rating": "{:.2f}"}),
                use_container_width=True,
                height=230,
            )
            st.caption("💡 **Score = predicted star rating (1–5)** derived from similar users via cosine-weighted collaborative filtering.")

    elif method == "Content-Based":
        rec = cb_model.recommend(user_id, ratings, products, n=5, category_ids=cat_filter)
        rec = enrich_recommendations(rec, "cosine_similarity")

        st.markdown('<p class="section-title">Top 5 Recommendations — Content-Based Filtering</p>', unsafe_allow_html=True)
        if rec.empty:
            st.warning("No recommendations found for this user with the current filters.")
        else:
            st.dataframe(
                rec.style.format({"price": "${:.2f}", "cosine_similarity": "{:.4f}"}),
                use_container_width=True,
                height=230,
            )
            st.caption("💡 **Score = cosine similarity** (0–1) between your TF-IDF taste profile and each product. Higher = closer match.")

    else:  # Compare Both
        st.markdown('<p class="section-title">Side-by-Side Comparison</p>', unsafe_allow_html=True)
        rec_cf = cf_model.recommend(user_id, products, n=5, category_ids=cat_filter)
        rec_cb = cb_model.recommend(user_id, ratings, products, n=5, category_ids=cat_filter)
        rec_cf = enrich_recommendations(rec_cf, "predicted_rating")
        rec_cb = enrich_recommendations(rec_cb, "cosine_similarity")

        cb_diag = model.evaluate_content_similarity_for_user(user_id, ratings, products, cb_model, top_k=5)
        cos = cb_diag["mean_top_k_cosine"]
        cos_txt = f"{cos:.4f}" if cos == cos else "n/a"

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🤝 Collaborative Filtering** — predicted rating")
            if rec_cf.empty:
                st.warning("No CF results.")
            else:
                st.dataframe(
                    rec_cf.style.format({"price": "${:.2f}", "predicted_rating": "{:.2f}"}),
                    use_container_width=True,
                    height=230,
                )
        with c2:
            st.markdown("**🧠 Content-Based** — cosine similarity")
            if rec_cb.empty:
                st.warning("No content-based results.")
            else:
                st.dataframe(
                    rec_cb.style.format({"price": "${:.2f}", "cosine_similarity": "{:.4f}"}),
                    use_container_width=True,
                    height=230,
                )

        st.info(
            f"📐 CF hold-out **RMSE**: **{pipe['cf_metrics']['rmse']:.3f}** stars &nbsp;|&nbsp; "
            f"Mean top-5 content **cosine** for this user: **{cos_txt}**"
        )

    # Rating history
    st.divider()
    st.markdown('<p class="section-title">This User\'s Rating History (top 15)</p>', unsafe_allow_html=True)
    hist = (
        ratings[ratings["user_id"] == user_id]
        .merge(products[["product_id", "product_name", "price"]], on="product_id", how="left")
        .sort_values("rating", ascending=False)
        .head(15)
    )
    if hist.empty:
        st.info("No rating history found for this user.")
    else:
        st.dataframe(
            hist[["product_id", "product_name", "price", "rating"]].style.format({"price": "${:.2f}"}),
            use_container_width=True,
            height=300,
        )

# ── Tab 2 · Insights & Charts ──────────────────────────────────────────────
with tab_viz:
    st.markdown('<p class="section-title">Dataset Insights</p>', unsafe_allow_html=True)

    # Quick stats row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Products", f"{products['product_id'].nunique():,}")
    m2.metric("Total Users", f"{ratings['user_id'].nunique():,}")
    m3.metric("Total Ratings", f"{len(ratings):,}")
    m4.metric("Avg Rating", f"{ratings['rating'].mean():.2f} ⭐")

    st.divider()

    # Chart helper — consistent dark style
    def _dark_fig(figsize=(10, 4)):
        fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1d27")
        ax.set_facecolor("#1a1d27")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3250")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#c4b5fd")
        return fig, ax

    pop = (
        ratings.groupby("product_id")
        .agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .merge(products[["product_id", "product_name"]], on="product_id")
        .sort_values("rating_count", ascending=False)
        .head(15)
    )

    fig1, ax1 = _dark_fig()
    ax1.barh(
        pop["product_name"].str.slice(0, 28)[::-1],
        pop["rating_count"][::-1],
        color="#7c3aed",
    )
    ax1.set_xlabel("Number of Ratings")
    ax1.set_title("Most Popular Products (by rating count)")
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    act = ratings.groupby("user_id").size().reset_index(name="n_ratings")
    fig2, ax2 = _dark_fig()
    ax2.hist(act["n_ratings"], bins=20, color="#0ea5e9", edgecolor="#1a1d27")
    ax2.set_xlabel("Ratings per User")
    ax2.set_ylabel("Users")
    ax2.set_title("User Activity Distribution")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    fig3, ax3 = _dark_fig()
    ratings["rating"].value_counts().sort_index().plot(
        kind="bar", ax=ax3, color="#10b981", edgecolor="#1a1d27"
    )
    ax3.set_xlabel("Star Rating")
    ax3.set_ylabel("Count")
    ax3.set_title("Ratings Distribution")
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

# ── Tab 3 · Trending & Search ──────────────────────────────────────────────
with tab_trend:
    st.markdown('<p class="section-title">Trending Products</p>', unsafe_allow_html=True)
    trend = model.trending_products(ratings, products, top_n=12)
    trend_display = trend.merge(products[["product_id", "price"]], on="product_id", how="left")
    if "price" in trend_display.columns:
        st.dataframe(
            trend_display.style.format({"price": "${:.2f}", "avg_rating": "{:.2f}", "trend_score": "{:.3f}"}),
            use_container_width=True,
        )
    else:
        st.dataframe(trend, use_container_width=True)

    st.divider()
    st.markdown('<p class="section-title">Catalog Search</p>', unsafe_allow_html=True)
    if not str(search_q).strip():
        st.caption("Enter a search term in the sidebar to filter the catalog.")
    else:
        hits = model.search_products(products, search_q, limit=25)
        if hits.empty:
            st.warning(f'No products found matching "{search_q}".')
        else:
            st.dataframe(
                hits[["product_id", "product_name", "category_id", "price"]].style.format({"price": "${:.2f}"}),
                use_container_width=True,
            )
            st.caption(f"Showing {len(hits)} result(s) for **{search_q}**.")
