import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RNG = np.random.default_rng(42)

N_USERS = 500
N_MOVIES = 300
N_RATINGS = 40000

GENRE_POOL = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Fantasy", "Horror", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Family",
]


def generate_movies(n_movies: int) -> pd.DataFrame:
    rows = []
    for movie_id in range(1, n_movies + 1):
        n_genres = RNG.integers(1, 4)
        genres = RNG.choice(GENRE_POOL, size=n_genres, replace=False)
        year = RNG.integers(1970, 2024)
        title = f"Movie {movie_id} ({year})"
        rows.append(
            {
                "movieId": movie_id,
                "title": title,
                "genres": "|".join(genres),
                "year": year,
                "_quality": RNG.normal(0, 1),
                "_popularity": RNG.exponential(1.0),
            }
        )
    return pd.DataFrame(rows)


def generate_users(n_users: int) -> pd.DataFrame:
    rows = []
    for user_id in range(1, n_users + 1):
        genre_affinity = {g: RNG.normal(0, 1) for g in GENRE_POOL}
        rows.append(
            {
                "userId": user_id,
                "_generosity": RNG.normal(0, 0.5),
                "_genre_affinity": genre_affinity,
            }
        )
    return pd.DataFrame(rows)


def generate_ratings(users: pd.DataFrame, movies: pd.DataFrame, n_ratings: int) -> pd.DataFrame:
    movie_probs = movies["_popularity"] / movies["_popularity"].sum()
    user_probs = np.ones(len(users)) / len(users)

    sampled_movie_idx = RNG.choice(len(movies), size=n_ratings, p=movie_probs)
    sampled_user_idx = RNG.choice(len(users), size=n_ratings, p=user_probs)

    start = datetime(2015, 1, 1)
    rows = []
    seen_pairs = set()
    for u_idx, m_idx in zip(sampled_user_idx, sampled_movie_idx):
        user = users.iloc[u_idx]
        movie = movies.iloc[m_idx]
        pair = (user["userId"], movie["movieId"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        genre_score = np.mean(
            [user["_genre_affinity"][g] for g in movie["genres"].split("|")]
        )
        raw_score = (
            3.3
            + 0.6 * movie["_quality"]
            + 0.5 * genre_score
            + user["_generosity"]
            + RNG.normal(0, 0.4)
        )
        rating = np.clip(round(raw_score * 2) / 2, 0.5, 5.0)

        timestamp = start + timedelta(minutes=int(RNG.integers(0, 60 * 24 * 365 * 4)))
        rows.append(
            {
                "userId": int(user["userId"]),
                "movieId": int(movie["movieId"]),
                "rating": float(rating),
                "timestamp": int(timestamp.timestamp()),
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def main():
    users = generate_users(N_USERS)
    movies = generate_movies(N_MOVIES)
    ratings = generate_ratings(users, movies, N_RATINGS)

    movies[["movieId", "title", "genres", "year"]].to_csv(
        "data/movies.csv", index=False
    )
    ratings.to_csv("data/ratings.csv", index=False)

    print(f"Generated {len(movies)} movies, {len(users)} users, {len(ratings)} ratings")
    print(f"Ratings sparsity: {len(ratings) / (len(users) * len(movies)):.2%}")


if __name__ == "__main__":
    main()
