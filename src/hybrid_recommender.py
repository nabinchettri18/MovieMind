import pandas as pd


class HybridRecommender:

    def __init__(
        self,
        movies,
        content_model,
        svd_model,
        content_weight=0.4,
        collaborative_weight=0.6,
    ):
        self.movies = movies.reset_index(drop=True).copy()
        self.content_model = content_model
        self.svd_model = svd_model
        self.content_weight = content_weight
        self.collaborative_weight = collaborative_weight
        self.ratings_lookup = {}

    def attach_ratings(self, ratings):
        self.ratings_lookup = (
            ratings.groupby("userId")["movieId"]
            .apply(lambda x: set(x.astype(int)))
            .to_dict()
        )

    def recommend(
        self,
        user_id,
        movie_title,
        num_recommendations=10,
    ):
        if movie_title not in self.content_model.title_to_index:
            return self.movies.head(0).copy()

        source_idx = self.content_model.title_to_index[
            movie_title
        ]

        content_scores = (
            self.content_model.matrix
            @ self.content_model.matrix[source_idx].T
        ).toarray().ravel()

        rated = set(
            self.ratings_lookup.get(int(user_id), set())
        )

        source_id = int(
            self.movies.iloc[source_idx]["movieId"]
        )

        rated.add(source_id)

        candidate_count = min(
            max(num_recommendations * 20, 100),
            len(self.movies) - 1,
        )

        candidate_indices = (
            content_scores.argsort()[-candidate_count:][::-1]
        )

        candidates = self.movies.iloc[
            candidate_indices
        ].copy()

        candidates = candidates[
            ~candidates["movieId"].isin(rated)
        ].copy()

        if candidates.empty:
            return candidates

        candidates["content_score"] = candidates[
            "movieId"
        ].map(
            lambda movie_id:
                self.content_model.score_movie(
                    movie_title,
                    int(movie_id),
                )
        )

        candidates["predicted_rating"] = candidates[
            "movieId"
        ].map(
            lambda movie_id:
                self.svd_model.predict(
                    user_id,
                    int(movie_id),
                )
        )

        candidates["svd_score"] = (
            (candidates["predicted_rating"] - 0.5) / 4.5
        ).clip(0, 1)

        candidates["hybrid_score"] = (
            self.collaborative_weight
            * candidates["svd_score"]
            + self.content_weight
            * candidates["content_score"]
        )

        return (
            candidates
            .sort_values(
                "hybrid_score",
                ascending=False,
            )
            .head(num_recommendations)
            .reset_index(drop=True)
        )