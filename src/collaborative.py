import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


class CollaborativeRecommender:

    def __init__(self, ratings, movies):

        self.ratings = ratings.copy()
        self.movies = movies.copy()

        # User × Movie rating matrix
        self.user_movie_matrix = self.ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating",
            fill_value=0
        )

        # Sparse matrix for memory efficiency
        self.sparse_matrix = csr_matrix(
            self.user_movie_matrix.values
        )

        # Similar-user model
        self.model = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=11
        )

        self.model.fit(self.sparse_matrix)

    def recommend(self, user_id, num_recommendations=10):

        if user_id not in self.user_movie_matrix.index:
            raise ValueError(
                f"User ID {user_id} not found."
            )

        # Locate user
        user_index = self.user_movie_matrix.index.get_loc(
            user_id
        )

        user_vector = self.sparse_matrix[user_index]

        # Find nearest users
        distances, indices = self.model.kneighbors(
            user_vector,
            n_neighbors=11
        )

        distances = distances.flatten()
        indices = indices.flatten()

        # Convert cosine distance → similarity
        similarities = 1 - distances

        # Remove the user themselves
        similar_users = []

        for i, similarity in zip(indices, similarities):

            if i == user_index:
                continue

            similar_user_id = self.user_movie_matrix.index[i]

            similar_users.append(
                (similar_user_id, similarity)
            )

        # Movies already watched
        watched_movies = set(
            self.ratings[
                self.ratings["userId"] == user_id
            ]["movieId"]
        )

        # Calculate weighted scores
        scores = {}

        for similar_user_id, similarity in similar_users:

            if similarity <= 0:
                continue

            user_ratings = self.ratings[
                self.ratings["userId"] == similar_user_id
            ]

            for _, row in user_ratings.iterrows():

                movie_id = row["movieId"]

                # Don't recommend watched movies
                if movie_id in watched_movies:
                    continue

                rating = row["rating"]

                if movie_id not in scores:
                    scores[movie_id] = {
                        "weighted_rating": 0,
                        "similarity_sum": 0
                    }

                scores[movie_id]["weighted_rating"] += (
                    similarity * rating
                )

                scores[movie_id]["similarity_sum"] += (
                    similarity
                )

        # Convert weighted scores into predictions
        predictions = []

        for movie_id, values in scores.items():

            if values["similarity_sum"] == 0:
                continue

            predicted_rating = (
                values["weighted_rating"]
                / values["similarity_sum"]
            )

            predictions.append(
                {
                    "movieId": movie_id,
                    "predicted_rating": predicted_rating
                }
            )

        recommendations = pd.DataFrame(predictions)

        if recommendations.empty:
            return recommendations

        # Sort by predicted rating
        recommendations = recommendations.sort_values(
            "predicted_rating",
            ascending=False
        ).head(num_recommendations)

        # Add movie information
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