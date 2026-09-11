import numpy as np
import pandas as pd


def rmse_mae(model, test: pd.DataFrame):
    preds = np.array([model.predict(u, m) for u, m in zip(test.userId, test.movieId)])
    actual = test["rating"].values
    rmse = np.sqrt(np.mean((preds - actual) ** 2))
    mae = np.mean(np.abs(preds - actual))
    return rmse, mae


def _ndcg_at_k(ranked_relevant_flags, k):
    dcg = sum(
        rel / np.log2(idx + 2) for idx, rel in enumerate(ranked_relevant_flags[:k])
    )
    ideal = sorted(ranked_relevant_flags, reverse=True)
    idcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def ranking_metrics(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    all_movie_ids,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 200,
):
    relevant_by_user = (
        test[test["rating"] >= relevance_threshold].groupby("userId")["movieId"].apply(set)
    )
    seen_by_user = train.groupby("userId")["movieId"].apply(set)

    precisions, recalls, ndcgs = [], [], []
    recommended_items_all = set()

    eval_users = list(relevant_by_user.index)[:max_users]
    for user_id in eval_users:
        relevant = relevant_by_user.get(user_id, set())
        if not relevant:
            continue
        already_seen = seen_by_user.get(user_id, set())
        candidates = [m for m in all_movie_ids if m not in already_seen]

        top_k = model.recommend(user_id, candidates, k=k)
        recommended_items_all.update(top_k)

        hits = [1 if m in relevant else 0 for m in top_k]
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant))
        ndcgs.append(_ndcg_at_k(hits, k))

    coverage = len(recommended_items_all) / len(all_movie_ids)

    return {
        "precision@k": np.mean(precisions) if precisions else 0.0,
        "recall@k": np.mean(recalls) if recalls else 0.0,
        "ndcg@k": np.mean(ndcgs) if ndcgs else 0.0,
        "catalog_coverage": coverage,
        "n_users_evaluated": len(precisions),
    }


def evaluate_model(name, model, train, test, all_movie_ids, k=10):
    rmse, mae = rmse_mae(model, test)
    rank_metrics = ranking_metrics(model, train, test, all_movie_ids, k=k)
    return {
        "model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        f"Precision@{k}": round(rank_metrics["precision@k"], 4),
        f"Recall@{k}": round(rank_metrics["recall@k"], 4),
        f"NDCG@{k}": round(rank_metrics["ndcg@k"], 4),
        "Coverage": round(rank_metrics["catalog_coverage"], 4),
    }
