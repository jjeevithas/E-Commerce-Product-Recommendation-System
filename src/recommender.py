import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ProductRecommender:
    """Hybrid recommender: collaborative SVD + content-based TF-IDF."""

    def __init__(self, products, interactions, n_components=20):
        self.products = products.copy()
        self.products["product_id"] = self.products["product_id"].astype(str)
        self.interactions = interactions.copy()
        self.interactions["product_id"] = self.interactions["product_id"].astype(str)

        self.product_ids = self.products["product_id"].tolist()
        self.product_index = {pid: i for i, pid in enumerate(self.product_ids)}

        # Content model
        text = (
            self.products["category"].fillna("") + " " +
            self.products["subcategory"].fillna("") + " " +
            self.products["brand"].fillna("") + " " +
            self.products["description"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(text)
        self.content_similarity = cosine_similarity(self.tfidf_matrix)

        # Collaborative model
        pivot = self.interactions.pivot_table(
            index="user_id",
            columns="product_id",
            values="weight",
            aggfunc="sum",
            fill_value=0
        )
        pivot = pivot.reindex(columns=self.product_ids, fill_value=0)

        self.user_ids = pivot.index.astype(str).tolist()
        self.user_index = {uid: i for i, uid in enumerate(self.user_ids)}
        self.user_item_matrix = pivot.values.astype(float)

        if self.user_item_matrix.shape[0] >= 2 and self.user_item_matrix.shape[1] >= 2:
            max_components = min(self.user_item_matrix.shape) - 1
            components = max(1, min(n_components, max_components))
            self.svd = TruncatedSVD(n_components=components, random_state=42)
            user_factors = self.svd.fit_transform(self.user_item_matrix)
            self.cf_predictions = user_factors @ self.svd.components_
        else:
            self.svd = None
            self.cf_predictions = self.user_item_matrix.copy()

        self.popularity = (
            self.interactions.groupby("product_id")["weight"]
            .sum()
            .reindex(self.product_ids, fill_value=0)
            .values.astype(float)
        )

    @staticmethod
    def _normalize(values):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        lo, hi = values.min(), values.max()
        if np.isclose(lo, hi):
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)

    def _content_scores(self, user_id):
        scores = np.zeros(len(self.product_ids))
        history = self.interactions[self.interactions["user_id"].astype(str) == str(user_id)]

        for _, row in history.iterrows():
            pid = str(row["product_id"])
            if pid in self.product_index:
                scores += float(row["weight"]) * self.content_similarity[self.product_index[pid]]

        return self._normalize(scores)

    def recommend(self, user_id, top_n=10, alpha=0.70):
        user_id = str(user_id)

        if user_id in self.user_index:
            cf_scores = self._normalize(self.cf_predictions[self.user_index[user_id]])
            content_scores = self._content_scores(user_id)
            scores = alpha * cf_scores + (1 - alpha) * content_scores
            seen = set(
                self.interactions.loc[
                    self.interactions["user_id"].astype(str) == user_id,
                    "product_id"
                ].astype(str)
            )
        else:
            scores = self._normalize(self.popularity)
            seen = set()

        result = self.products.copy()
        result["score"] = scores
        result = result[~result["product_id"].isin(seen)]
        return result.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    def similar_products(self, product_id, top_n=5):
        product_id = str(product_id)
        if product_id not in self.product_index:
            return pd.DataFrame()

        idx = self.product_index[product_id]
        similarities = self.content_similarity[idx]
        order = np.argsort(similarities)[::-1]

        selected = []
        for i in order:
            if self.product_ids[i] != product_id:
                selected.append((self.product_ids[i], similarities[i]))
            if len(selected) >= top_n:
                break

        score_map = dict(selected)
        result = self.products[self.products["product_id"].isin(score_map)].copy()
        result["similarity"] = result["product_id"].map(score_map)
        return result.sort_values("similarity", ascending=False).reset_index(drop=True)
