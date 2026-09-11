import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, liked_threshold: float = 4.0):
        self.liked_threshold = liked_threshold

    def fit(self, train: pd.DataFrame, movies: pd.DataFrame):
        self.movies_ = movies.reset_index(drop=True)
        genre_text = self.movies_["genres"].str.replace("|", " ", regex=False)

        self.vectorizer_ = TfidfVectorizer(token_pattern=r"[A-Za-z\-]+")
        self.item_vectors_ = self.vectorizer_.fit_transform(genre_text)

        self.movie_id_to_row_ = {
            mid: row for row, mid in enumerate(self.movies_["movieId"])
        }
        self.global_mean_ = train["rating"].mean()

        self.user_vectors_ = {}
        liked = train[train["rating"] >= self.liked_threshold]
        for user_id, group in liked.groupby("userId"):
            rows = [
                self.movie_id_to_row_[m]
                for m in group["movieId"]
                if m in self.movie_id_to_row_
            ]
            if rows:
                self.user_vectors_[user_id] = np.asarray(
                    self.item_vectors_[rows].mean(axis=0)
                )
        return self

    def predict(self, user_id, movie_id) -> float:
        if user_id not in self.user_vectors_ or movie_id not in self.movie_id_to_row_:
            return self.global_mean_
        sim = cosine_similarity(
            self.user_vectors_[user_id],
            self.item_vectors_[self.movie_id_to_row_[movie_id]],
        )[0][0]
        return float(np.clip(1.0 + sim * 4.0, 0.5, 5.0))

    def recommend(self, user_id, candidate_movie_ids, k=10):
        scores = {m: self.predict(user_id, m) for m in candidate_movie_ids}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in ranked[:k]]

    def has_taste_profile(self, user_id) -> bool:
        return user_id in self.user_vectors_
