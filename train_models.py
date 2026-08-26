from pathlib import Path
import joblib

from src.data_loader import (
    load_movies,
    load_ratings,
    load_tags,
)

from src.content_based import (
    ContentBasedRecommender,
)

from src.svd_recommender import (
    SVDRecommender,
)

from src.hybrid_recommender import (
    HybridRecommender,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_FILE = MODEL_DIR / "moviemind_models.joblib"


print("=" * 60)
print("MovieMind 32M - ONE TIME MODEL BUILD")
print("=" * 60)

print("\n[1/5] Loading movies...")
movies = load_movies()
print(f"Loaded {len(movies):,} movies.")

print("\n[2/5] Loading tags...")
tags = load_tags()
print(f"Loaded {len(tags):,} tags.")

print("\n[3/5] Loading ratings...")
ratings = load_ratings()
print(f"Loaded {len(ratings):,} ratings.")

print("\n[4/5] Building recommendation engine...")

print("  • Content model: genres + tags")
content_model = ContentBasedRecommender(
    movies,
    tags,
)

print("  • Collaborative model")
print(
    "    Using 5,000,000 ratings for the training sample."
)

svd_model = SVDRecommender(
    ratings,
    movies,
    factors=50,
    max_training_ratings=5_000_000,
)

print("  • Hybrid model")
hybrid_model = HybridRecommender(
    movies,
    content_model,
    svd_model,
    content_weight=0.4,
    collaborative_weight=0.6,
)

hybrid_model.attach_ratings(ratings)

print("\n[5/5] Saving MovieMind engine...")

joblib.dump(
    {
        "movies": movies,
        "ratings": ratings[
            ["userId", "movieId", "rating"]
        ],
        "hybrid_model": hybrid_model,
        "dataset": "MovieLens 32M",
        "version": 3,
    },
    MODEL_FILE,
    compress=3,
)

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"Model saved to:\n{MODEL_FILE}")
print("\nNow run:")
print("streamlit run app.py")