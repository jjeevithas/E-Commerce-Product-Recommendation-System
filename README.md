# 🛍️ E-Commerce Product Recommendation System

An e-commerce analytics and machine learning project that analyzes user interactions and product information to generate personalized product recommendations.

The project uses user behavior such as views, cart events, and purchases to understand product preferences and generate relevant recommendations.

## 🚀 Live Demo

👉 [View the live recommendation system](https://e-commerce-appuct-recommendation-system-fpg7bmfe2vsuewbxhkkjct.streamlit.app/)

## 🎯 Project Goal

The main goal of this project is to analyze e-commerce user behavior and build a recommendation system that suggests relevant products to users.

The project combines data analysis, recommendation techniques, and machine learning.

It uses:

* Collaborative Filtering
* Matrix Factorization using TruncatedSVD
* Content-Based Filtering using TF-IDF and Cosine Similarity
* Hybrid recommendation scoring
* Streamlit for the interactive web application
* Time-based evaluation using Precision@10, Recall@10, and NDCG@10

## 🛠️ Technology Stack

* **Python** — application development and data analysis
* **Pandas** — data processing and analysis
* **Scikit-learn** — machine learning algorithms
* **TruncatedSVD** — matrix factorization
* **TF-IDF** — product feature representation
* **Cosine Similarity** — product similarity
* **Streamlit** — interactive web application
* **CSV** — dataset storage

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

### 1. User Behavior Analysis

The system uses user-product interaction data to understand how users interact with products.

The main interaction types are:

* View
* Cart
* Purchase

Different weights are assigned to these actions:

* View = 1
* Cart = 3
* Purchase = 5

This gives more importance to stronger actions, such as adding a product to the cart or purchasing it.

### 2. Collaborative Filtering

A user-product interaction matrix is created from the weighted user activities.

**TruncatedSVD** is then applied to the interaction matrix to reduce its dimensions and identify hidden patterns between users and products.

This helps recommend products based on similar user behavior.

### 3. Content-Based Filtering

The system also uses product information such as:

* Category
* Subcategory
* Brand
* Description

This information is converted into **TF-IDF vectors**.

**Cosine similarity** is then used to find products that are similar based on their characteristics.

### 4. Hybrid Recommendation

The final recommendation combines collaborative filtering and content-based filtering:

```text
Final Score = alpha × Collaborative Score
            + (1 - alpha) × Content Score
```

The Streamlit application allows the collaborative filtering weight to be adjusted, which changes the balance between user-behavior-based and content-based recommendations.

## 📊 Model Evaluation

To evaluate the recommendation system, I used a **time-based train/test split**.

This is important because using future interactions during training can lead to data leakage.

For each user:

1. Interactions are sorted by timestamp.
2. Earlier interactions are used for training.
3. The latest interactions are kept for testing.
4. The recommendation model is trained using only the training data.
5. Recommendations are generated for the users.
6. The recommendations are compared with the held-out test products.

The system is evaluated using:

* **Precision@10**
* **Recall@10**
* **NDCG@10**

These metrics help measure how relevant the recommended products are and how well the system ranks the products.

## 📁 Dataset

The project uses a synthetic e-commerce dataset containing:

* 80 users
* Multiple product categories
* Product information and metadata
* Timestamped user interactions
* View, cart, and purchase events

The dataset is mainly used for demonstrating the recommendation workflow and can be replaced with a larger real-world dataset in the future.

## 💡 What the Project Demonstrates

This project demonstrates how e-commerce interaction data can be used to:

* Understand user product preferences
* Analyze different types of user interactions
* Identify relationships between users and products
* Find similar products
* Generate personalized recommendations
* Evaluate recommendation quality using standard metrics

## 🔮 Future Improvements

Some improvements I would like to add later:

* Use a larger real-world e-commerce dataset
* Add more user and product features
* Experiment with deep learning recommendation models
* Improve recommendation ranking
* Add user authentication
* Store user interactions in a database
* Add more interactive analytics to the Streamlit application

## 👩‍💻 Author

**Jeevitha S**

Developed an e-commerce analytics and machine learning project using **Python, Pandas, Scikit-learn, and Streamlit** to analyze user behavior and generate personalized product recommendations.
