from src.data_loader import load_movies, load_ratings
from src.collaborative import CollaborativeRecommender


movies = load_movies()
ratings = load_ratings()

recommender = CollaborativeRecommender(
    ratings,
    movies
)

results = recommender.recommend(
    user_id=1,
    num_recommendations=10
)

print("\nPersonalized recommendations for User 1:\n")

for _, movie in results.iterrows():
    print(
        f"{movie['title']} "
        f"| {movie['genres']} "
        f"| Predicted Rating: "
        f"{movie['predicted_rating']:.2f}"
    )