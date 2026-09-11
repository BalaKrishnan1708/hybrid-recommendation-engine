import numpy as np
import pandas as pd


class MatrixFactorization:
    def __init__(
        self,
        n_factors: int = 30,
        n_epochs: int = 20,
        lr: float = 0.005,
        reg: float = 0.02,
        seed: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.rng = np.random.default_rng(seed)

    def fit(self, train: pd.DataFrame, verbose: bool = False):
        self.global_mean_ = train["rating"].mean()

        self.user_ids_ = train["userId"].unique()
        self.item_ids_ = train["movieId"].unique()
        self.user_idx_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.item_idx_ = {m: i for i, m in enumerate(self.item_ids_)}

        n_users, n_items = len(self.user_ids_), len(self.item_ids_)
        self.P = self.rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = self.rng.normal(0, 0.1, (n_items, self.n_factors))
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        rows = list(
            zip(
                train["userId"].map(self.user_idx_),
                train["movieId"].map(self.item_idx_),
                train["rating"],
            )
        )

        for epoch in range(self.n_epochs):
            self.rng.shuffle(rows)
            sq_err_sum = 0.0
            for u, i, r in rows:
                pred = self.global_mean_ + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
                err = r - pred

                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                p_u_old = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * p_u_old - self.reg * self.Q[i])

                sq_err_sum += err**2

            if verbose:
                rmse = np.sqrt(sq_err_sum / len(rows))
                print(f"  epoch {epoch + 1}/{self.n_epochs}  train RMSE={rmse:.4f}")
        return self

    def predict(self, user_id, movie_id) -> float:
        if user_id not in self.user_idx_ or movie_id not in self.item_idx_:
            return self.global_mean_
        u, i = self.user_idx_[user_id], self.item_idx_[movie_id]
        pred = self.global_mean_ + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]
        return float(np.clip(pred, 0.5, 5.0))

    def recommend(self, user_id, candidate_movie_ids, k=10):
        scores = {m: self.predict(user_id, m) for m in candidate_movie_ids}
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [m for m, _ in ranked[:k]]

    def is_known_user(self, user_id) -> bool:
        return user_id in self.user_idx_
