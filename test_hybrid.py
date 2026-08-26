from src.data_loader import load_movies, load_ratings
from src.content_based import ContentBasedRecommender
from src.svd_recommender import SVDRecommender
from src.hybrid_recommender import HybridRecommender


movies = load_movies()
ratings = load_ratings()

print("Loading content-based model...")

content_model = ContentBasedRecommender(
    movies
)

print("Loading SVD model...")

svd_model = SVDRecommender(
    ratings,
    movies,
    factors=50
)

print("Creating hybrid model...")

hybrid_model = HybridRecommender(
    movies,
    content_model,
    svd_model,
    content_weight=0.4,
    collaborative_weight=0.6
)

print("Hybrid model ready!")

results = hybrid_model.recommend(
    user_id=1,
    movie_title="Toy Story (1995)",
    num_recommendations=10
)

print("\nMovieMind recommendations:\n")

for _, movie in results.iterrows():

    print(
        f"{movie['title']} "
        f"| {movie['genres']} "
        f"| SVD: {movie['svd_score']:.3f} "
        f"| Content: {movie['content_score']:.3f} "
        f"| Hybrid: {movie['hybrid_score']:.3f}"
    )