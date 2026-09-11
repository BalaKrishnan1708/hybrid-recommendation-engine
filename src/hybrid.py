import numpy as np


class HybridRecommender:
    def __init__(self, cf_model, content_model, alpha: float = 0.7):
        self.cf_model = cf_model
        self.content_model = content_model
        self.alpha = alpha

    def predict(self, user_id, movie_id) -> float:
        if not self.cf_model.is_known_user(user_id):
            return self.content_model.predict(user_id, movie_id)

        cf_score = self.cf_model.predict(user_id, movie_id)
        content_score = self.content_model.predict(user_id, movie_id)
        return self.alpha * cf_score + (1 - self.alpha) * content_score

    def recommend(self, user_id, candidate_movie_ids, k=10):
        scores = {m: self.predict(user_id, m) for m in candidate_movie_ids}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in ranked[:k]]
