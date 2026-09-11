import numpy as np
import pandas as pd


class GlobalMeanBaseline:
    def fit(self, train: pd.DataFrame):
        self.global_mean_ = train["rating"].mean()
        return self

    def predict(self, user_id, movie_id) -> float:
        return self.global_mean_

    def recommend(self, user_id, candidate_movie_ids, k=10):
        return list(candidate_movie_ids[:k])


class ItemMeanBaseline:
    def fit(self, train: pd.DataFrame):
        self.global_mean_ = train["rating"].mean()
        self.item_means_ = train.groupby("movieId")["rating"].mean()
        self.item_counts_ = train.groupby("movieId")["rating"].count()
        return self

    def predict(self, user_id, movie_id) -> float:
        return self.item_means_.get(movie_id, self.global_mean_)

    def recommend(self, user_id, candidate_movie_ids, k=10):
        scores = {
            m: self.item_means_.get(m, self.global_mean_) for m in candidate_movie_ids
        }
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in ranked[:k]]
