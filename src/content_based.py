import numpy as np
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer


class ContentBasedRecommender:

    def __init__(self, movies, tags=None):
        self.movies = movies.reset_index(drop=True).copy()
        self.tags = tags
        self.vectorizer = None
        self.matrix = None
        self.title_to_index = {}
        self._fit()

    def _fit(self):
        tag_map = {}

        if self.tags is not None and not self.tags.empty:
            grouped = (
                self.tags.groupby("movieId")["tag"]
                .apply(
                    lambda x: " ".join(
                        str(v).lower().replace("|", " ")
                        for v in x
                        if str(v).strip()
                    )
                )
            )
            tag_map = grouped.to_dict()

        documents = []

        for _, movie in self.movies.iterrows():
            genres = str(movie["genres"]).replace("|", " ")
            tags = tag_map.get(int(movie["movieId"]), "")
            documents.append(f"{genres} {genres} {tags}")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=120_000,
            sublinear_tf=True,
        )

        self.matrix = normalize(
            self.vectorizer.fit_transform(documents)
        )

        self.title_to_index = {
            str(title): i
            for i, title in enumerate(self.movies["title"])
        }

    def similar_movies(self, movie_title, num_recommendations=10):
        if movie_title not in self.title_to_index:
            return self.movies.head(0).copy()

        idx = self.title_to_index[movie_title]

        scores = np.asarray(
            (self.matrix @ self.matrix[idx].T).toarray()
        ).ravel()

        scores[idx] = -1

        count = min(num_recommendations, len(scores) - 1)

        if count <= 0:
            return self.movies.head(0).copy()

        indices = np.argpartition(scores, -count)[-count:]
        indices = indices[np.argsort(scores[indices])[::-1]]

        result = self.movies.iloc[indices].copy()
        result["content_score"] = scores[indices]

        return result.reset_index(drop=True)

    def score_movie(self, source_title, movie_id):
        if source_title not in self.title_to_index:
            return 0.0

        matches = self.movies.index[
            self.movies["movieId"] == movie_id
        ]

        if len(matches) == 0:
            return 0.0

        source_idx = self.title_to_index[source_title]
        target_idx = int(matches[0])

        score = (
            self.matrix[source_idx]
            @ self.matrix[target_idx].T
        )

        return float(score.toarray()[0, 0])