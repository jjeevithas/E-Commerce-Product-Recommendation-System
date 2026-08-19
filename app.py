from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.recommender import ProductRecommender
from src.evaluate import evaluate_model

st.set_page_config(
    page_title="E-Commerce Product Recommendation System",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ E-Commerce Product Recommendation System")
st.caption("Hybrid Model: Collaborative Filtering + SVD Matrix Factorization + Content-Based Filtering")

@st.cache_data
def load_data():
    products = pd.read_csv(ROOT / "data" / "products.csv")
    interactions = pd.read_csv(ROOT / "data" / "interactions.csv", parse_dates=["timestamp"])
    return products, interactions

@st.cache_resource
def build_model(products, interactions):
    return ProductRecommender(products, interactions)

products, interactions = load_data()
model = build_model(products, interactions)

with st.sidebar:
    st.header("Controls")
    user_id = st.selectbox("Select User", sorted(interactions["user_id"].unique()))
    top_n = st.slider("Recommendations", 5, 15, 10)
    alpha = st.slider(
        "Collaborative Filtering Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.70,
        step=0.05
    )
    st.caption("Higher value gives more importance to collaborative filtering.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🎯 Recommendations", "🔎 Similar Products", "📊 Evaluation", "📁 Dataset"]
)

with tab1:
    st.subheader(f"Personalized recommendations for {user_id}")
    recommendations = model.recommend(user_id, top_n=top_n, alpha=alpha)

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.dataframe(
            recommendations[
                ["product_name", "category", "subcategory", "brand", "price", "score"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.bar_chart(
            recommendations.set_index("product_name")["score"]
        )

with tab2:
    selected_name = st.selectbox(
        "Choose a product",
        products["product_name"].tolist()
    )
    selected_id = products.loc[
        products["product_name"] == selected_name, "product_id"
    ].iloc[0]

    similar = model.similar_products(selected_id, top_n=8)
    st.subheader(f"Products similar to: {selected_name}")
    st.dataframe(
        similar[
            ["product_name", "category", "subcategory", "brand", "price", "similarity"]
        ],
        use_container_width=True,
        hide_index=True
    )

with tab3:
    st.subheader("Correct Time-Based Evaluation")
    st.write(
        "The model is trained only on earlier interactions. "
        "Newer held-out interactions are used as the test set."
    )

    metrics, train, test, per_user = evaluate_model(
        products,
        interactions,
        k=10,
        alpha=alpha
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@10", f"{metrics['precision']:.3f}")
    c2.metric("Recall@10", f"{metrics['recall']:.3f}")
    c3.metric("NDCG@10", f"{metrics['ndcg']:.3f}")

    st.caption(
        f"Training interactions: {len(train)} | "
        f"Test interactions: {len(test)} | "
        f"Evaluated users: {len(per_user)}"
    )

    if not per_user.empty:
        st.dataframe(per_user, use_container_width=True, hide_index=True)

with tab4:
    c1, c2 = st.columns(2)
    c1.metric("Users", interactions["user_id"].nunique())
    c2.metric("Products", products["product_id"].nunique())

    st.subheader("Interaction Types")
    st.bar_chart(interactions["event"].value_counts())

    st.subheader("Product Categories")
    st.bar_chart(products["category"].value_counts())

st.divider()
st.caption("Built with Python, Pandas, NumPy, Scikit-learn and Streamlit.")
