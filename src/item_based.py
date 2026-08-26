import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class ItemBasedRecommender:

    def __init__(self, ratings, movies):

        self.ratings = ratings.copy()
        self.movies = movies.copy()

        # Movie × User matrix
        self.movie_user_matrix = self.ratings.pivot_table(
            index="movieId",
            columns="userId",
            values="rating",
            fill_value=0
        )

        # Sparse representation
        self.sparse_matrix = csr_matrix(
            self.movie_user_matrix.values
        )

        # Find similar movies
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=11
        )

        self.model.fit(self.sparse_matrix)

        self.movie_indices = pd.Series(
            self.movie_user_matrix.index,
            index=range(len(self.movie_user_matrix))
        )

    def recommend(self, user_id, num_recommendations=10):

        # Movies rated by the user
        user_ratings = self.ratings[
            self.ratings["userId"] == user_id
        ].copy()

        if user_ratings.empty:
            raise ValueError(
                f"User ID {user_id} has no ratings."
            )

        # Only use movies the user rated highly
        liked_movies = user_ratings[
            user_ratings["rating"] >= 4
        ].sort_values(
            "rating",
            ascending=False
        )

        scores = {}
        weights = {}

        for _, row in liked_movies.iterrows():

            movie_id = row["movieId"]
            user_rating = row["rating"]

            if movie_id not in self.movie_user_matrix.index:
                continue

            movie_index = self.movie_user_matrix.index.get_loc(
                movie_id
            )

            movie_vector = self.sparse_matrix[movie_index]

            distances, indices = self.model.kneighbors(
                movie_vector,
                n_neighbors=11
            )

            distances = distances.flatten()
            indices = indices.flatten()

            for distance, similar_index in zip(
                distances[1:],
                indices[1:]
            ):

                similar_movie_id = (
                    self.movie_user_matrix.index[
                        similar_index
                    ]
                )

                similarity = 1 - distance

                if similarity <= 0:
                    continue

                # Don't recommend something already rated
                if similar_movie_id in set(
                    user_ratings["movieId"]
                ):
                    continue

                weighted_score = (
                    similarity * user_rating
                )

                scores[similar_movie_id] = (
                    scores.get(similar_movie_id, 0)
                    + weighted_score
                )

                weights[similar_movie_id] = (
                    weights.get(similar_movie_id, 0)
                    + similarity
                )

        if not scores:
            return pd.DataFrame()

        predictions = []

        for movie_id in scores:

            if weights[movie_id] == 0:
                continue

            predicted_rating = (
                scores[movie_id]
                / weights[movie_id]
            )

            predictions.append(
                {
                    "movieId": movie_id,
                    "predicted_rating": predicted_rating
                }
            )

        recommendations = pd.DataFrame(predictions)

        recommendations = recommendations.sort_values(
            "predicted_rating",
            ascending=False
        ).head(num_recommendations)

        recommendations = recommendations.merge(
            self.movies[
                ["movieId", "title", "genres"]
            ],
            on="movieId",
            how="left"
        )

        return recommendations[
            [
                "movieId",
                "title",
                "genres",
                "predicted_rating"
            ]
        ]