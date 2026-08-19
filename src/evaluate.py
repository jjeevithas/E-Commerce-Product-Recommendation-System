import numpy as np
import pandas as pd

from src.recommender import ProductRecommender


def precision_recall_ndcg_at_k(recommended, relevant, k=10):
    """Calculate ranking metrics for one user."""
    recommended = list(map(str, recommended))[:k]
    relevant = set(map(str, relevant))

    if not relevant:
        return 0.0, 0.0, 0.0

    hits = [item for item in recommended if item in relevant]
    precision = len(hits) / k
    recall = len(hits) / len(relevant)

    dcg = 0.0
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            dcg += 1 / np.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


def time_based_split(interactions, test_size=0.20):
    """
    For every user, keep the newest interactions as test data.
    Training data contains only earlier interactions.
    """
    data = interactions.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])

    train_parts = []
    test_parts = []

    for _, group in data.sort_values("timestamp").groupby("user_id"):
        n = len(group)
        n_test = max(1, int(np.ceil(n * test_size)))
        n_test = min(n_test, n - 2)  # retain enough training history

        if n_test < 1:
            continue

        train_parts.append(group.iloc[:-n_test])
        test_parts.append(group.iloc[-n_test:])

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)
    return train, test


def evaluate_model(products, interactions, k=10, alpha=0.70):
    """
    Correct evaluation:
    1. Split interactions chronologically.
    2. Build the recommender ONLY with training interactions.
    3. Recommend unseen products.
    4. Compare recommendations with held-out test products.
    """
    train, test = time_based_split(interactions)

    model = ProductRecommender(products, train)

    rows = []
    for user_id, group in test.groupby("user_id"):
        relevant = group["product_id"].astype(str).tolist()
        if not relevant:
            continue

        recommended = model.recommend(
            user_id=user_id,
            top_n=k,
            alpha=alpha
        )["product_id"].astype(str).tolist()

        p, r, n = precision_recall_ndcg_at_k(recommended, relevant, k)
        rows.append({
            "user_id": user_id,
            "precision": p,
            "recall": r,
            "ndcg": n
        })

    results = pd.DataFrame(rows)
    if results.empty:
        metrics = {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    else:
        metrics = {
            "precision": float(results["precision"].mean()),
            "recall": float(results["recall"].mean()),
            "ndcg": float(results["ndcg"].mean())
        }

    return metrics, train, test, results
