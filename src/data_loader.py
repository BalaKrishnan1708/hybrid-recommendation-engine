import pandas as pd


def load_data(data_dir: str = "data"):
    movies = pd.read_csv(f"{data_dir}/movies.csv")
    ratings = pd.read_csv(f"{data_dir}/ratings.csv")
    return movies, ratings


def time_based_split(ratings: pd.DataFrame, test_frac: float = 0.2):
    ratings_sorted = ratings.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(ratings_sorted) * (1 - test_frac))
    train = ratings_sorted.iloc[:split_idx].reset_index(drop=True)
    test = ratings_sorted.iloc[split_idx:].reset_index(drop=True)

    train_users, train_items = set(train.userId), set(train.movieId)
    test_cold_users = set(test.userId) - train_users
    test_cold_items = set(test.movieId) - train_items
    print(
        f"Train: {len(train)} ratings | Test: {len(test)} ratings | "
        f"Cold-start users in test: {len(test_cold_users)} | "
        f"Cold-start items in test: {len(test_cold_items)}"
    )
    return train, test
