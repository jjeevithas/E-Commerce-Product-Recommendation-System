# 🛍️ E-Commerce Product Recommendation System

A machine learning project I built to recommend products based on user interactions and product information.

The system combines different recommendation techniques to suggest products that are more relevant to each user.

## 🚀 Live Demo

👉 [View the live recommendation system](https://e-commerce-appuct-recommendation-system-fpg7bmfe2vsuewbxhkkjct.streamlit.app/)

## 🎯 Project Goal

The main goal of this project is to build a recommendation system that can suggest products based on user behavior and product details.

The project uses:

* Collaborative Filtering
* Matrix Factorization using TruncatedSVD
* Content-Based Filtering using TF-IDF and Cosine Similarity
* Hybrid recommendation scoring
* Streamlit for the web application
* Time-based evaluation using Precision@K, Recall@K, and NDCG@K

## 📂 Project Structure

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

## ⚙️ Installation

### 1. Open the project folder

```bash
cd Ecommerce_Product_Recommendation_System_Corrected
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## 🧠 How the Recommendation System Works

### Collaborative Filtering

First, the system creates a user-product interaction matrix using the user's activities.

Different actions are given different weights:

* View = 1
* Cart = 3
* Purchase = 5

This helps give more importance to stronger user actions, such as adding a product to the cart or purchasing it.

I then use **TruncatedSVD** to reduce the interaction matrix and learn hidden patterns between users and products.

### Content-Based Filtering

The system also looks at the information available for each product.

It uses:

* Category
* Subcategory
* Brand
* Description

This information is converted into **TF-IDF vectors**, and **cosine similarity** is used to find products with similar characteristics.

### 🔄 Hybrid Recommendation

The final recommendation combines both approaches:

```text
Final Score = alpha × Collaborative Score
            + (1 - alpha) × Content Score
```

The Streamlit application allows the collaborative filtering weight to be adjusted, so the recommendation can be changed between behavior-based and content-based results.

## 📊 Model Evaluation

I used a **time-based train/test split** to evaluate the recommendation system and avoid data leakage.

For each user:

1. Interactions are sorted by time.
2. Older interactions are used for training.
3. The latest interactions are kept for testing.
4. The model is trained only using the training data.
5. Recommendations are generated for the users.
6. The recommendations are compared with the held-out test products.

The model is evaluated using:

* **Precision@10**
* **Recall@10**
* **NDCG@10**

This approach makes the evaluation more realistic because future interactions are not used while training the model.

## 📁 Dataset

The project uses a synthetic dataset containing:

* 80 users
* Multiple product categories
* Product details and metadata
* Timestamped user interactions
* View, cart, and purchase events

The CSV files can be replaced with a larger real-world dataset in the future.

## 🔮 Future Improvements

Some improvements I would like to add later:

* Use a larger real-world e-commerce dataset
* Add more user and product features
* Experiment with deep learning recommendation models
* Improve recommendation ranking
* Add user authentication
* Store user interactions in a database
* Deploy the application online

  ## 👩‍💻 Author

**Jeevitha S**

Developed a machine learning-based product recommendation system using **Python, Scikit-learn, and Streamlit**, combining user behavior and product information to generate relevant recommendations.


Precision@10, Recall@10, and NDCG@10.
