import sys
import pandas as pd

sys.path.insert(0, "src")

from data_loader import load_data, time_based_split
from baseline import GlobalMeanBaseline, ItemMeanBaseline
from collaborative_filtering import MatrixFactorization
from content_based import ContentBasedRecommender
from hybrid import HybridRecommender
from evaluate import evaluate_model

K = 10


def main():
    print("=" * 60)
    print("RECOMMENDATION ENGINE — training & evaluation pipeline")
    print("=" * 60)

    movies, ratings = load_data("data")
    all_movie_ids = movies["movieId"].tolist()

    print(f"\nLoaded {len(movies)} movies, {ratings['userId'].nunique()} users, "
          f"{len(ratings)} ratings\n")

    train, test = time_based_split(ratings, test_frac=0.2)

    print("\nTraining models...")

    global_mean = GlobalMeanBaseline().fit(train)
    item_mean = ItemMeanBaseline().fit(train)

    print("  fitting matrix factorization (SGD)...")
    mf = MatrixFactorization(n_factors=30, n_epochs=15, lr=0.007, reg=0.03)
    mf.fit(train, verbose=False)

    print("  fitting content-based (TF-IDF over genres)...")
    content = ContentBasedRecommender(liked_threshold=4.0).fit(train, movies)

    hybrid = HybridRecommender(cf_model=mf, content_model=content, alpha=0.7)

    print("\nEvaluating on held-out (future) ratings...\n")
    results = [
        evaluate_model("Global Mean (floor)", global_mean, train, test, all_movie_ids, k=K),
        evaluate_model("Item Mean (popularity)", item_mean, train, test, all_movie_ids, k=K),
        evaluate_model("Matrix Factorization (CF)", mf, train, test, all_movie_ids, k=K),
        evaluate_model("Content-Based (TF-IDF)", content, train, test, all_movie_ids, k=K),
        evaluate_model("Hybrid (CF + Content)", hybrid, train, test, all_movie_ids, k=K),
    ]

    results_df = pd.DataFrame(results).set_index("model")
    print(results_df.to_string())
    results_df.to_csv("results_comparison.csv")
    print("\nSaved results_comparison.csv")

    print("\n" + "=" * 60)
    print("SAMPLE RECOMMENDATIONS")
    print("=" * 60)
    movie_titles = movies.set_index("movieId")["title"]

    sample_users = train["userId"].drop_duplicates().sample(2, random_state=1).tolist()
    for user_id in sample_users:
        already_seen = set(train[train["userId"] == user_id]["movieId"])
        candidates = [m for m in all_movie_ids if m not in already_seen]

        print(f"\n--- User {user_id} ---")
        top_rated = (
            train[train["userId"] == user_id]
            .sort_values("rating", ascending=False)
            .head(3)
        )
        print("Previously liked:")
        for _, row in top_rated.iterrows():
            print(f"  {movie_titles[row.movieId]}  (rated {row.rating})")

        print(f"Top {5} hybrid recommendations:")
        for movie_id in hybrid.recommend(user_id, candidates, k=5):
            print(f"  {movie_titles[movie_id]}")


if __name__ == "__main__":
    main()
