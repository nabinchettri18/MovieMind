from src.data_loader import load_movies
from src.content_based import ContentBasedRecommender


movies = load_movies()

recommender = ContentBasedRecommender(movies)

results = recommender.recommend(
    "Toy Story (1995)",
    num_recommendations=10
)

print("\nMovies similar to Toy Story:\n")

for _, movie in results.iterrows():
    print(
        f"{movie['title']} "
        f"| {movie['genres']} "
        f"| Similarity: {movie['similarity_score']:.3f}"
    )