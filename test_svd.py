from src.data_loader import load_movies, load_ratings
from src.svd_recommender import SVDRecommender


movies = load_movies()
ratings = load_ratings()

print("Training SVD model...")

recommender = SVDRecommender(
    ratings,
    movies,
    factors=50
)

print("Model trained successfully!")

results = recommender.recommend(
    user_id=1,
    num_recommendations=10
)

print("\nSVD recommendations for User 1:\n")

for _, movie in results.iterrows():

    print(
        f"{movie['title']} "
        f"| {movie['genres']} "
        f"| Predicted Rating: "
        f"{movie['predicted_rating']:.2f}"
    )