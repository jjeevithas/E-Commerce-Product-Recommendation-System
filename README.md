# 🛍️ E-Commerce Product Recommendation System

A portfolio-ready machine learning project that recommends products based on user behavior and product information.

## Project Goal

Build a recommendation system that suggests relevant products using:

- Collaborative Filtering
- Matrix Factorization using TruncatedSVD
- Content-Based Filtering using TF-IDF and Cosine Similarity
- Hybrid recommendation scoring
- Streamlit web application
- Correct time-based evaluation using Precision@K, Recall@K and NDCG@K

## Project Structure

```text
Ecommerce_Product_Recommendation_System_Corrected/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── products.csv
│   └── interactions.csv
│
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   └── evaluate.py
│
└── notebooks/
    └── analysis.py
```

## Installation

### 1. Open the project folder

```bash
cd Ecommerce_Product_Recommendation_System_Corrected
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

### 3. Install packages

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

## How the Model Works

### Collaborative Filtering

A user-product interaction matrix is created from views, cart events and purchases.

Weights:

- View = 1
- Cart = 3
- Purchase = 5

TruncatedSVD performs matrix factorization to learn latent user and product patterns.

### Content-Based Filtering

Product category, subcategory, brand and description are converted into TF-IDF vectors. Cosine similarity finds products with similar content.

### Hybrid Recommendation

The final score is:

```text
Final Score = alpha × Collaborative Score
            + (1 - alpha) × Content Score
```

The Streamlit sidebar allows the collaborative weight to be adjusted.

## Correct Evaluation Method

The previous version had a data leakage problem. This version fixes it.

For each user:

1. Sort interactions by time.
2. Keep earlier interactions for training.
3. Hold out the newest interactions for testing.
4. Train the model only on training data.
5. Generate recommendations.
6. Compare recommendations against held-out test products.

Metrics:

- Precision@10
- Recall@10
- NDCG@10

This prevents the test products from being included in the user's training history.

## Dataset

The project includes a synthetic dataset with:

- 80 users
- Multiple product categories
- Product metadata
- Hundreds of timestamped interactions
- View, cart and purchase events

You can later replace the CSV files with a larger real-world dataset.

## Resume Description

**E-Commerce Product Recommendation System | Python, Scikit-learn, Streamlit**

Developed a hybrid product recommendation system using collaborative filtering, TruncatedSVD matrix factorization, and TF-IDF content-based filtering. Built an interactive Streamlit dashboard for personalized and similar-product recommendations and evaluated the model using time-based train/test splitting with Precision@K, Recall@K, and NDCG@K.
