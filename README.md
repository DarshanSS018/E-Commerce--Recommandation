# E-Commerce Product Recommendation System

A small, end-to-end data science project that recommends products using **user-based collaborative filtering** and **content-based filtering** (TF-IDF + cosine similarity). It includes preprocessing, model evaluation, charts, and a **Streamlit** web dashboard with search, category filters, and trending products.

---

## Author

**Darshan S S**

---

## Problem Statement

Online stores have large catalogs; shoppers benefit from suggestions that reflect both **what similar users liked** (collaborative signal) and **what matches the text and price profile of items they enjoyed** (content signal). This project connects a real **category hierarchy** (`data/category_tree.csv`) with **synthetic products and ratings** so you can experiment locally without sharing private user data.

---

## Algorithms Used

### 1. User-Based Collaborative Filtering
Builds a user × product ratings matrix. For each pair of users, cosine similarity is computed on **co-rated** items (with ratings mean-centered per user). To recommend, similar neighbors' centered ratings are combined as a **weighted prediction** for items the target user has not yet rated. Top items are ranked by predicted rating (1–5 scale).

### 2. Content-Based Filtering
Each product is represented by **TF-IDF** features on `product_name` + `description`, plus a **normalized price** column. A **user profile** is the rating-weighted average of the TF-IDF vectors for products that user has rated. Unseen products are scored by **cosine similarity** between the user profile and each product's vector.

### 3. Evaluation
- **Collaborative Filtering:** RMSE on a random 20 % hold-out of ratings (lower is better).
- **Content-Based:** Mean cosine similarity of the top-5 recommended items for the selected user (higher means the suggestions are closer to the taste profile in vector space).

---

## Project Structure

```
E-Commerce Recommendation System/
├── app.py                        # Streamlit web dashboard
├── model.py                      # Preprocessing, recommenders, metrics, helpers
├── requirements.txt
├── README.md
├── scripts/
│   └── generate_sample_data.py   # Regenerates products.csv and ratings.csv
└── data/
    ├── category_tree.csv          # Category hierarchy (categoryid, parentid)
    ├── products.csv               # Sample product catalog
    └── ratings.csv                # Sample user–item ratings
```

---

## Technologies

| Layer | Library |
|---|---|
| Data handling | `pandas`, `numpy` |
| ML / feature engineering | `scikit-learn` (TF-IDF, cosine similarity, train-test split, RMSE, MinMaxScaler, OneHotEncoder) |
| Sparse matrices | `scipy` |
| Visualization | `matplotlib` |
| Web UI | `streamlit` |

---

## How to Run

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the app

Run from the project root (the folder that contains `app.py`):

```bash
streamlit run app.py
```

If `streamlit` is not found on your PATH, use:

```bash
python -m streamlit run app.py
```

### 4. Use the dashboard

Open the URL shown in the terminal (usually `http://localhost:8501`).

- Select a **User ID** in the sidebar.
- Choose a recommendation method: **Collaborative Filtering**, **Content-Based**, or **Compare Both**.
- Optionally filter by **category ID** or enter a **search term** to filter the catalog.
- Switch to the **Insights & Charts** tab to view dataset statistics and distribution charts.
- Switch to the **Trending & Search** tab to explore trending products or search the catalog.

---

## Regenerating Sample Data

The sample `products.csv` and `ratings.csv` are generated from the category tree so that every `category_id` in the product catalog exists in `category_tree.csv`. To rebuild them with different random seeds, run from the project root:

```bash
python scripts/generate_sample_data.py
```

---

## Notes for Learners

- **Missing values:** Empty `parentid` in the category tree is filled with `-1`. Missing prices are imputed with the median. Invalid rating rows are dropped.
- **Categorical → numeric:** Categories are one-hot encoded during preprocessing (available for optional extensions). The content model uses TF-IDF text features plus scaled price.
- **Normalization:** `MinMaxScaler` scales price to [0, 1] for use in the content-based feature vectors.
- **Scores in the UI:**
  - Collaborative recommendations show **predicted rating (1–5)**.
  - Content-based recommendations show **cosine similarity (0–1)**; higher means a closer match to the user's taste profile.

---

## Dataset Note

The **category hierarchy** (`category_tree.csv`) is sourced from a Retail Hero–style category tree. The **products and ratings data** (`products.csv`, `ratings.csv`) are **synthetically generated** for educational/demo purposes and do not represent real transactions or users.

---

## License

Educational demo project. The category tree source is the provided third-party file. Synthetic products and ratings are generated for learning purposes only. Third-party library licenses apply to their respective packages.
