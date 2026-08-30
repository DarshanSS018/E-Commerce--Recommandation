# E-commerce recommendation system (Python)

A small end-to-end data science project that recommends products using **user-based collaborative filtering** and **content-based filtering** (cosine similarity). It includes preprocessing, evaluation, charts, and a **Streamlit** web UI with search, category filters, and trending products.

## Problem statement

Online stores have large catalogs; shoppers benefit from suggestions that reflect both **what similar users liked** (collaborative signal) and **what matches the text/price profile of items they enjoyed** (content signal). This demo connects a real **category hierarchy** (`data/category_tree.csv` from your Retail Hero–style tree) with **synthetic products and ratings** so you can experiment locally without sharing private user data.

## Algorithms used

1. **User-based collaborative filtering**  
   Build a user × product ratings matrix. For each pair of users, cosine similarity is computed on **co-rated** items (with ratings centered by each user’s mean). To recommend, similar users’ centered ratings are combined to **predict** a star rating for items the target user has not rated. Top items are ranked by predicted rating.

2. **Content-based filtering**  
   Each product is represented by **TF–IDF** features on `product_name` + `description`, plus a **normalized price** column. A **user profile** is the rating-weighted average of the vectors for products that user rated. Unseen products are scored by **cosine similarity** between the user profile and each product vector.

3. **Evaluation**  
   - Collaborative: **RMSE** on a random 20% hold-out of ratings (train CF on the rest).  
   - Content: **Mean cosine similarity** of the top-5 recommended items for the selected user (higher means the suggestions are closer to the profile in vector space—useful for comparison in the UI, not a substitute for online A/B testing).

## Project structure

```
e commerce recommendation system/
├── app.py              # Streamlit UI
├── model.py            # Preprocessing + recommenders + metrics + helpers
├── requirements.txt
├── README.md
├── scripts/
│   └── generate_sample_data.py
└── data/
    ├── category_tree.csv   # Category hierarchy (categoryid, parentid)
    ├── products.csv        # Sample catalog (generated to link to real category IDs)
    └── ratings.csv         # Sample user–item ratings
```

## How to run

1. **Create a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the app** from the project root (the folder that contains `app.py`):

   ```bash
   streamlit run app.py
   ```

   If your terminal does not find `streamlit`, use:

   ```bash
   python -m streamlit run app.py
   ```

4. Open the URL shown in the terminal (usually `http://localhost:8501`). Enter a **User ID**, choose a method or **Compare both**, and optionally narrow by **category** or use **search** on the *Trending & search* tab.

### Regenerating sample `products.csv` / `ratings.csv`

The sample catalog and ratings are created so every `category_id` in `products.csv` exists in `category_tree.csv`. To rebuild them with different randomness, run from the project root:

```bash
python scripts/generate_sample_data.py
```

## Notes for learners

- **Missing values**: Empty `parentid` in the category tree is filled with `-1`. Missing prices (if any) are imputed with the median. Invalid rating rows are dropped.
- **Categorical → numeric**: Categories are **one-hot encoded** in preprocessing (available for extensions); the content model uses TF–IDF + scaled price.
- **Normalize**: `MinMaxScaler` normalizes **price** to \([0,1]\) for the content vectors.
- **Scores in the UI**: Collaborative recommendations show **predicted rating (1–5)**; content-based show **cosine similarity** in \([0,1]\) (approximately—depends on vector sparsity).

## Libraries

- `pandas`, `numpy` — data handling  
- `scikit-learn` — TF–IDF, cosine similarity, train/test split, RMSE, encoders/scalers  
- `matplotlib` — charts in Streamlit  
- `streamlit` — web UI  
- `scipy` — sparse matrix utilities for TF–IDF + numeric features  

## License

Educational demo; category tree source is your provided file; synthetic products/ratings are generated for learning.
