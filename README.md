# Recommendation Engine

A hybrid movie recommendation system implementing and comparing four
approaches: popularity baseline, collaborative filtering (matrix
factorization / SVD), content-based filtering (TF-IDF), and a hybrid model.

## Architecture

```
recommendation-engine/
├── data/
│   ├── generate_synthetic_data.py   # generates movies.csv / ratings.csv
│   ├── movies.csv                   # movieId, title, genres, year
│   └── ratings.csv                  # userId, movieId, rating, timestamp
├── src/
│   ├── data_loader.py               # loading + time-based train/test split
│   ├── baseline.py                  # GlobalMeanBaseline, ItemMeanBaseline
│   ├── collaborative_filtering.py   # MatrixFactorization (SGD, biased SVD)
│   ├── content_based.py             # TF-IDF + cosine similarity
│   ├── hybrid.py                    # switching + weighted blend
│   └── evaluate.py                  # RMSE/MAE, Precision@K, Recall@K, NDCG@K
├── main.py                          # runs training + evaluation + demo
└── results_comparison.csv           # generated after running main.py
```

### Why each piece exists

- **Baselines** (`baseline.py`): a popularity/global-mean predictor. Any
  real model must beat these, or the added complexity isn't worth it.
- **Collaborative filtering** (`collaborative_filtering.py`): biased
  matrix factorization trained with SGD — `r_hat = mu + b_u + b_i + p_u·q_i`.
  Learns latent taste dimensions purely from the ratings matrix. This is
  the same model family that won the Netflix Prize. Weak on cold-start
  (new users/movies with no ratings yet).
- **Content-based** (`content_based.py`): represents each movie as a
  TF-IDF vector over genres, and each user as the average vector of
  movies they rated highly. Works for brand-new movies (cold-start item)
  since it only needs metadata, not ratings history.
- **Hybrid** (`hybrid.py`): switches to content-based for cold-start
  users, otherwise blends CF and content scores (weighted 70/30 by
  default). This mirrors what production recommenders actually do.
- **Evaluation** (`evaluate.py`): rating-accuracy metrics (RMSE/MAE) plus
  ranking metrics (Precision@K, Recall@K, NDCG@K), plus catalog coverage
  — because a model that only ever recommends the same 10 popular movies
  can look accurate while being a bad recommender.

## Running it

```bash
pip install numpy pandas scikit-learn
python data/generate_synthetic_data.py   # only needed once
python main.py
```

This prints a metrics comparison table across all five models, saves it
to `results_comparison.csv`, and prints sample recommendations for two
users.

## Using real MovieLens data instead of synthetic data

The synthetic generator exists only because this environment has no
internet access to download the real dataset. To use real data:

1. Download `ml-latest-small.zip` (or the full 25M version) from
   https://grouplens.org/datasets/movielens/latest/
2. Copy `movies.csv` and `ratings.csv` into this project's `data/`
   folder — the schema already matches.
3. Run `python main.py` — no code changes needed.

## Results on the bundled synthetic dataset

| Model | RMSE ↓ | MAE ↓ | Precision@10 ↑ | Recall@10 ↑ | NDCG@10 ↑ | Coverage ↑ |
|---|---|---|---|---|---|---|
| Global Mean | 0.93 | 0.76 | 0.006 | 0.012 | 0.023 | 0.05 |
| Item Mean (popularity) | 0.75 | 0.61 | 0.043 | 0.132 | 0.169 | 0.05 |
| Matrix Factorization | 0.57 | 0.45 | 0.047 | 0.134 | 0.226 | 0.13 |
| Content-Based | 1.57 | 1.35 | 0.027 | 0.066 | 0.108 | 0.91 |
| **Hybrid** | 0.70 | 0.57 | **0.048** | **0.145** | 0.215 | 0.39 |

Read: pure CF wins on rating accuracy but recommends from a narrow slice
of the catalog (13% coverage). Content-based is inaccurate on ratings but
recommends broadly (91% coverage) since it doesn't need interaction data.
The hybrid gets the best ranking quality (Precision/Recall) while nearly
tripling CF's catalog coverage — the accuracy/diversity trade-off a real
hybrid system is built to manage.

Exact numbers will differ slightly with real MovieLens data and are
randomized by the synthetic generator's seed, but the ordering and
trade-offs should hold.

## Extending this project

- **Cold-start demo**: add a brand-new movie with genres but zero
  ratings, and show CF returns the global mean for it while content-based
  and the hybrid return a meaningful score.
- **Implicit feedback**: switch from explicit 0.5–5 star ratings to
  implicit signals (watched/clicked) and use an ALS-style implicit
  matrix factorization instead.
- **Deep learning**: replace `MatrixFactorization` with a two-tower
  neural network (separate user and item embedding towers trained
  jointly) for a more "modern" architecture.
- **Serving**: wrap `HybridRecommender.recommend()` in a small
  Flask/FastAPI endpoint for a live demo, or build a Streamlit UI on
  top of `main.py`'s pipeline.
