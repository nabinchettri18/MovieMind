import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class SVDRecommender:

    def __init__(
        self,
        ratings,
        movies,
        factors=50,
        max_training_ratings=5_000_000,
        random_state=42,
    ):
        self.movies = movies.reset_index(drop=True).copy()
        self.factors = factors
        self.max_training_ratings = max_training_ratings
        self.random_state = random_state

        self.user_index = {}
        self.movie_index = {}
        self.user_factors = None
        self.movie_factors = None
        self.global_mean = 3.5

        self._fit(ratings)

    def _fit(self, ratings):
        ratings = ratings[
            ["userId", "movieId", "rating"]
        ].dropna()

        if len(ratings) > self.max_training_ratings:
            ratings = ratings.sample(
                n=self.max_training_ratings,
                random_state=self.random_state,
            )

        users = ratings["userId"].unique()
        items = ratings["movieId"].unique()

        self.user_index = {
            int(v): i for i, v in enumerate(users)
        }

        self.movie_index = {
            int(v): i for i, v in enumerate(items)
        }

        rows = ratings["userId"].map(
            self.user_index
        ).to_numpy()

        cols = ratings["movieId"].map(
            self.movie_index
        ).to_numpy()

        values = ratings["rating"].astype(
            np.float32
        ).to_numpy()

        matrix = csr_matrix(
            (values, (rows, cols)),
            shape=(len(users), len(items)),
            dtype=np.float32,
        )

        self.global_mean = float(values.mean())

        n_components = min(
            self.factors,
            max(2, min(matrix.shape) - 1),
        )

        svd = TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            n_iter=5,
            random_state=self.random_state,
        )

        self.user_factors = svd.fit_transform(
            matrix
        ).astype(np.float32)

        self.movie_factors = svd.components_.T.astype(
            np.float32
        )

    def predict(self, user_id, movie_id):
        ui = self.user_index.get(int(user_id))
        mi = self.movie_index.get(int(movie_id))

        if ui is None or mi is None:
            return self.global_mean

        value = (
            self.global_mean
            + float(
                np.dot(
                    self.user_factors[ui],
                    self.movie_factors[mi],
                )
            )
        )

        return float(np.clip(value, 0.5, 5.0))